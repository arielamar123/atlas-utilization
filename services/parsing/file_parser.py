"""
FileParser service - Single responsibility: Parse ROOT files.

Extracts events from ATLAS ROOT files using uproot.
No orchestration logic, no state management.
"""

import logging
import awkward as ak
import numpy as np
import uproot
import itertools
from contextlib import nullcontext
from typing import Iterable, Optional

from services.parsing import schemas
from services import consts


class FileParser:
    """
    Service for parsing individual ROOT files.
    
    Pure function-like service with no state. All methods are static.
    """
    
    @staticmethod
    def parse_file(
        file_path: str,
        tree_names: list[str],
        release_year: str,
        batch_size: int = 40_000,
        enable_jet_tagging: bool = False,
        jet_btagging_thresholds: Optional[dict[str, float]] = None,
        read_timeout_sec: float = 300.0,
        objects_to_parse: Optional[Iterable[str]] = None,
        remote_read_semaphore=None,
        remote_serial_read_min_entries: int = 100_000,
    ) -> Optional[ak.Array]:
        """
        Parse a single ROOT file and return events.
        
        Args:
            file_path: Path or URI to ROOT file
            tree_names: List of possible tree names to search for
            release_year: Release year identifier (e.g., "2024r-pp")
            batch_size: Number of entries to process per batch
            
        Returns:
            Awkward array of events with particle objects, or None if parsing failed
        """
        try:
            # The timeout is retained by uproot's XRootD source and applies to
            # open and chunk-read requests, preventing an unavailable remote
            # file from occupying a parsing worker forever.
            with uproot.open(file_path, timeout=read_timeout_sec) as root_file:
                return FileParser._parse_opened_file(
                    root_file,
                    tree_names,
                    release_year,
                    batch_size,
                    file_path,
                    enable_jet_tagging,
                    jet_btagging_thresholds,
                    objects_to_parse,
                    remote_read_semaphore,
                    remote_serial_read_min_entries,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to parse file {file_path}") from e
    
    @staticmethod
    def _parse_opened_file(
        root_file,
        tree_names: list[str],
        release_year: str,
        batch_size: int,
        file_path: str,
        enable_jet_tagging: bool,
        jet_btagging_thresholds: Optional[dict[str, float]],
        objects_to_parse: Optional[Iterable[str]] = None,
        remote_read_semaphore=None,
        remote_serial_read_min_entries: int = 100_000,
    ) -> Optional[ak.Array]:
        """Parse an already-opened ROOT file."""
        tree_name = FileParser._get_data_tree_name(root_file.keys(), tree_names)
        tree = root_file[tree_name]
        all_tree_branches = set(tree.keys())
        n_entries = tree.num_entries
        
        obj_branches = FileParser._extract_branches_by_schema(
            all_tree_branches,
            release_year
        )

        obj_branches = FileParser._restrict_branches_to_requested_objects(
            obj_branches,
            objects_to_parse,
            enable_jet_tagging,
        )

        # Auxiliary tagger branches are large and are irrelevant when tagging
        # is disabled. Avoid reading them at all in that mode.
        if not enable_jet_tagging:
            obj_branches.pop("DirectObjects", None)

        if not obj_branches:
            logging.warning(f"No particles found in schema for file {file_path}")
            return None
        
        obj_branches = FileParser._filter_accessible_branches(tree, obj_branches)
        
        if not obj_branches:
            logging.warning(f"No accessible particles found in file {file_path}")
            return None
        
        all_branches = set(itertools.chain.from_iterable(obj_branches.values()))
        is_large_remote_read = FileParser._should_limit_remote_read(
            file_path,
            n_entries,
            remote_serial_read_min_entries,
        )
        read_guard = (
            remote_read_semaphore
            if is_large_remote_read and remote_read_semaphore is not None
            else nullcontext()
        )
        with read_guard:
            obj_events = FileParser._read_file_in_batches(
                tree,
                all_branches,
                obj_branches,
                n_entries,
                batch_size
            )
        obj_events = FileParser._split_combined_leptons(obj_events, release_year)
        should_calculate_jet_tags = (
            enable_jet_tagging
            and "Jets" in obj_events
            and "DirectObjects" in obj_events
        )
        if should_calculate_jet_tags:
            obj_events = FileParser._calculate_btagging_and_split(obj_events, jet_btagging_thresholds)
        # Strip out DirectObjects -- they are not physics objects!
        obj_events.pop("DirectObjects", None)
        if objects_to_parse is not None:
            requested = set(objects_to_parse)
            obj_events = {
                name: values
                for name, values in obj_events.items()
                if name in requested
            }
        return ak.zip(obj_events, depth_limit=1)

    @staticmethod
    def _should_limit_remote_read(
        file_path: str,
        n_entries: int,
        minimum_entries: int,
    ) -> bool:
        is_remote = file_path.startswith(("root://", "http://", "https://"))
        return is_remote and n_entries >= minimum_entries

    @staticmethod
    def _restrict_branches_to_requested_objects(
        obj_branches: dict[str, dict[str, str]],
        objects_to_parse: Optional[Iterable[str]],
        enable_jet_tagging: bool,
    ) -> dict[str, dict[str, str]]:
        """Prune remote branches before any accessibility test or full read."""
        if objects_to_parse is None:
            return obj_branches

        requested = set(objects_to_parse)
        source_objects = requested - {"BJets"}
        needs_jet_classification = bool(
            enable_jet_tagging and requested.intersection({"Jets", "BJets"})
        )
        if "BJets" in requested:
            source_objects.add("Jets")
        if needs_jet_classification:
            source_objects.add("DirectObjects")

        restricted = {}
        for object_name, branch_mapping in obj_branches.items():
            if object_name not in source_objects:
                continue
            restricted[object_name] = {
                branch: quantity
                for branch, quantity in branch_mapping.items()
                if not (
                    quantity == "mass"
                    and object_name in schemas.FIXED_MASS_OBJECTS
                )
            }
        return restricted

    @staticmethod
    def _split_combined_leptons(
        obj_events: dict[str, ak.Array], release_year: str
    ) -> dict[str, ak.Array]:
        """Split legacy ``lep_*`` branches using their absolute PDG type."""
        normalized_release = schemas.normalize_release_year(release_year)
        if normalized_release not in {"2016e-8tev", "2025e-13tev-beta"}:
            return obj_events
        source = obj_events.get("Electrons")
        if source is None:
            source = obj_events.get("Muons")
        if source is None or "type" not in source.fields:
            raise ValueError(
                f"{normalized_release} combined lepton branches require lep_type"
            )
        abs_type = abs(source["type"])
        obj_events["Electrons"] = source[abs_type == 11]
        obj_events["Muons"] = source[abs_type == 13]
        return obj_events

    @staticmethod
    def _calculate_btagging_and_split(
        obj_events: dict[str, ak.Array],
        jet_btagging_thresholds: Optional[dict[str, float]]
    ) -> dict[str, ak.Array]:
        """
        For each Jet objects in each event, calculates the b-tagging discriminant.
        Then, decides if each jet is a bjet or not, using the per-algorithm threshold in `jet_btagging_thresholds`.
        If bjet, stores in obj_events["BJet"] and removes from obj_events["Jets"]. Otherwise, leaves the Jet be.
        """
        # TODO Find a cleaner way to determine if dealing with nanoAOD, PHYSLITE, etc.
        # TODO Maybe add an explicit check if the algorithm-specific threshold exists in the configuration dict before
        # accessing it. However, I'd rather fail parsing then give false physics data.
        if "Jets" not in obj_events:
            raise ValueError("Jet tagging was enabled but no Jets collection was parsed")
        if "DirectObjects" not in obj_events:
            raise ValueError("Jet tagging was enabled but tagger branches are missing")
        if not jet_btagging_thresholds:
            raise ValueError("Jet tagging was enabled without tagger thresholds")

        if "Jet_btagDeepFlavB" in obj_events["DirectObjects"].fields:
            # CMS: Discriminant is pre-calculated as the Jet_btagDeepFlavB field. Can change to a different algorithm if needed.
            # See https://cms-opendata-workshop.github.io/workshop2024-lesson-physics-objects/instructor/05-btagging.html
            if "btagDeepFlavB" not in jet_btagging_thresholds:
                raise ValueError("Missing btagDeepFlavB threshold for CMS jets")
            is_bjet = (
                obj_events["DirectObjects"]["Jet_btagDeepFlavB"]
                > jet_btagging_thresholds["btagDeepFlavB"]
            )
        elif "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb" in obj_events["DirectObjects"].fields:
            # ATLAS
            if "DL1d" not in jet_btagging_thresholds:
                raise ValueError("Missing DL1d threshold for ATLAS jets")
            indices = obj_events["DirectObjects"]["AnalysisJetsAuxDyn.btaggingLink/AnalysisJetsAuxDyn.btaggingLink.m_persIndex"]
            pb = obj_events["DirectObjects"]["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb"][indices]
            pc = obj_events["DirectObjects"]["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc"][indices]
            pu = obj_events["DirectObjects"]["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu"][indices]
            # DL1d score: log(pb / (fc*pc + (1-fc)*pu)), fc=0.018 is standard ATLAS.
            fc = 0.018
            denominator = fc * pc + (1 - fc) * pu
            valid_ratio = (pb > 0) & (denominator > 0)
            # Avoid evaluating either division by zero or log(0). A positive
            # b probability with zero background probability has +inf score;
            # zero b probability is untagged with -inf score.
            safe_pb = ak.where(valid_ratio, pb, 1.0)
            safe_denominator = ak.where(valid_ratio, denominator, 1.0)
            dl1d = np.log(safe_pb / safe_denominator)
            dl1d = ak.where(
                (pb > 0) & (denominator <= 0),
                np.inf,
                dl1d,
            )
            dl1d = ak.where(pb <= 0, -np.inf, dl1d)
            is_bjet = dl1d > jet_btagging_thresholds["DL1d"]
        else:
            raise ValueError(
                "Jet tagging was enabled but no supported tagger discriminant was parsed"
            )
        obj_events["BJets"] = obj_events["Jets"][is_bjet]
        obj_events["Jets"] = obj_events["Jets"][~is_bjet]
        return obj_events
    
    @staticmethod
    def _get_data_tree_name(
        root_file_keys: list[str],
        possible_tree_names: list[str]
    ) -> str:
        if not possible_tree_names:
            return "CollectionTree"
        
        available_trees = [key[:-2] if key.endswith(';1') else key for key in root_file_keys]
        
        for tree_name in possible_tree_names:
            if tree_name in available_trees:
                return tree_name
        
        return "CollectionTree"
    
    @staticmethod
    def _extract_branches_by_schema(
        tree_branches: set[str],
        release_year: str
    ) -> dict[str, dict[str, str]]:
        """
        Extract branches by object based on release-specific schema.
        
        Returns:
            Dict mapping object names to their branch mappings
            Format: {obj_name: {full_branch: quantity, ...}}
        """
        try:
            record_id = None
            if release_year.startswith("record_"):
                try:
                    record_id = int(release_year.split("_")[1])
                except (ValueError, IndexError):
                    pass
            
            schema_config = schemas.get_schema_for_release(release_year, record_id=record_id)
        except KeyError:
            logging.warning(
                f"Release year '{release_year}' not found in schemas. "
                "Attempting auto-detection."
            )
            return FileParser._auto_detect_branches(tree_branches)
        
        obj_branches = {}
        objects = schema_config["objects"]
        direct_objects = schema_config.get("direct_objects", [])
        naming_pattern = schema_config.get("naming_pattern", "dotted")
        
        for obj_name, fields in objects.items():
            if naming_pattern == "flat":
                obj_branches_for_obj = FileParser._extract_flat_branches(
                    obj_name, fields, tree_branches, release_year
                )
            else:
                obj_branches_for_obj = FileParser._extract_dotted_branches(
                    obj_name, fields, tree_branches, release_year, schema_config
                )
            
            if obj_branches_for_obj:
                obj_branches[obj_name] = obj_branches_for_obj
        # Keep direct object names as-is, but store them under the "DirectObjects" key.
        obj_branches.update({"DirectObjects": {k: k for k in direct_objects}})
        return obj_branches
    
    @staticmethod
    def _prepare_obj_branch_name(
        obj_name: str,
        release_year: str = "2024r-pp",
        field: str = None,
        record_id: int = None
    ) -> str:
        """
        Prepare object branch name using release-specific template.

        For flat naming with a field, returns "ObjectName_field".
        For flat naming without a field, returns just the mapped object name.
        For dotted naming, returns "PrefixObjectSuffix".
        Falls back to ATLAS default naming on unknown releases.
        """
        try:
            if release_year.startswith("record_") and record_id is None:
                try:
                    record_id = int(release_year.split("_")[1])
                except (ValueError, IndexError):
                    pass

            schema = schemas.get_schema_for_release(release_year, record_id=record_id)
            naming_pattern = schema.get("naming_pattern", "dotted")
            object_mappings = schema.get("object_mappings", {})
            branch_obj_name = object_mappings.get(obj_name, obj_name)

            if naming_pattern == "flat":
                if field:
                    return f"{branch_obj_name}_{field}"
                return branch_obj_name
            else:
                prefix = schema["branch_prefix"]
                suffix = schema["branch_suffix"]
                return f"{prefix}{branch_obj_name}{suffix}"
        except KeyError:
            logging.warning(f"Release year '{release_year}' not found. Using default branch naming.")
            return "Analysis" + obj_name + "AuxDyn"

    @staticmethod
    def _find_cms_branches(
        base_branch_name: str,
        fields: list[str],
        obj_field_paths: dict,
        tree_branches: set[str]
    ) -> dict[str, str]:
        """
        Find CMS-style nested branches for a given object.

        CMS branch structure: {base}/{base}obj/{base}obj.{field_path}
        """
        branch_mappings = {}
        available_fields = []

        base_obj = f"{base_branch_name}obj"
        obj_container_patterns = [
            base_obj,
            f"{base_branch_name}/{base_obj}",
        ]
        has_obj_container = any(
            pattern in branch
            for branch in tree_branches
            for pattern in obj_container_patterns
        )
        if not has_obj_container:
            return {}

        for field in fields:
            if field not in obj_field_paths:
                continue

            field_path = obj_field_paths[field]
            field_indicator = consts.CMS_FIELD_INDICATORS.get(field)
            if not field_indicator:
                continue

            field_path_suffix = field_path[4:] if field_path.startswith("obj.") else field_path
            expected_path = f"{base_branch_name}/{base_obj}/{base_obj}.{field_path_suffix}"

            if expected_path in tree_branches:
                branch_mappings[expected_path] = field
                available_fields.append(field)
                continue

            matching = [
                branch for branch in tree_branches
                if branch.startswith(base_branch_name)
                and f"{base_obj}/" in branch
                and field_indicator in branch
            ]
            if matching:
                full_path = max(matching, key=len)
                branch_mappings[full_path] = field
                available_fields.append(field)
                continue

            field_selection_path = f"{base_obj}.{field_path_suffix}"
            branch_mappings[field_selection_path] = field
            available_fields.append(field)

        if FileParser._can_calculate_inv_mass(available_fields):
            return branch_mappings
        return {}

    @staticmethod
    def _extract_flat_branches(
        obj_name: str,
        fields: list[str],
        tree_branches: set[str],
        release_year: str
    ) -> dict[str, str]:
        """Extract branches using flat naming pattern (object_field)."""
        branch_base = FileParser._prepare_obj_branch_name(obj_name, release_year=release_year)
        available_fields = [
            f for f in fields if f"{branch_base}_{f}" in tree_branches
        ]
        
        if FileParser._can_calculate_inv_mass(available_fields):
            return {
                f"{branch_base}_{field}": field
                for field in available_fields
            }
        
        return {}
    
    @staticmethod
    def _extract_dotted_branches(
        obj_name: str,
        fields: list[str],
        tree_branches: set[str],
        release_year: str,
        schema_config: dict
    ) -> dict[str, str]:
        """Extract branches using dotted naming pattern (object.field)."""
        branch_name = FileParser._prepare_obj_branch_name(obj_name, release_year=release_year)
        logging.debug(f"ATLAS-style naming for {obj_name}, branch base: {branch_name}")
        
        field_paths = schema_config.get("field_paths", {})
        obj_field_paths = field_paths.get(obj_name, {})
        
        if obj_field_paths:
            return FileParser._find_cms_branches(
                branch_name, fields, obj_field_paths, tree_branches
            )
        
        branch_to_quantity = {}
        available_fields = []
        
        for field in fields:
            branch_full = f"{branch_name}.{field}"
            if branch_full in tree_branches:
                available_fields.append(field)
                branch_to_quantity[branch_full] = field
            elif field == "mass":
                mass_branch = f"{branch_name}.m"
                if mass_branch in tree_branches:
                    available_fields.append(field)
                    branch_to_quantity[mass_branch] = field
        
        if FileParser._can_calculate_inv_mass(available_fields):
            return branch_to_quantity
        
        return {}
    
    @staticmethod
    def _can_calculate_inv_mass(
        available_fields: list[str],
        ref_system: set[str] = {'phi', 'eta', 'pt'}
    ) -> bool:
        return ref_system.issubset(set(available_fields))
    
    @staticmethod
    def _filter_accessible_branches(
        tree,
        obj_branches: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        """
        Test branch accessibility and filter out inaccessible ones.
        
        Reads ONE entry with ALL candidate branches at once to minimize
        HTTP round-trips for remote ROOT files.
        """
        all_candidate_branches = []
        for branch_mapping in obj_branches.values():
            all_candidate_branches.extend(branch_mapping.keys())
        
        accessible_set = set()
        try:
            test_arr = tree.arrays(
                all_candidate_branches,
                entry_start=0, entry_stop=1,
                library="ak"
            )
            accessible_set = set(test_arr.fields)
        except Exception:
            for branch_path in all_candidate_branches:
                try:
                    test_arr = tree.arrays(
                        branch_path,
                        entry_start=0, entry_stop=1,
                        library="ak"
                    )
                    if branch_path in test_arr.fields:
                        accessible_set.add(branch_path)
                except Exception:
                    continue
        
        accessible_obj_branches = {}
        for obj_name, branch_mapping in obj_branches.items():
            accessible_branches = {
                bp: qty for bp, qty in branch_mapping.items()
                if bp in accessible_set
            }
            if accessible_branches and FileParser._can_calculate_inv_mass(
                list(accessible_branches.values())
            ) or obj_name == "DirectObjects":
                accessible_obj_branches[obj_name] = accessible_branches
        
        return accessible_obj_branches
    
    @staticmethod
    def _read_file_in_batches(
        tree,
        all_branches: set[str],
        obj_branches: dict[str, dict[str, str]],
        n_entries: int,
        batch_size: int
    ) -> dict[str, ak.Array]:
        # The parser returns one complete file-level EventBatch, so retaining a
        # list of 40k-entry arrays and concatenating it afterward did not bound
        # memory. It only added remote round trips and a second full-size copy.
        # Keep ``batch_size`` in the API for compatibility with callers.
        del batch_size
        try:
            file_data = tree.arrays(
                all_branches,
                entry_start=0,
                entry_stop=n_entries,
                library="ak",
            )
        except Exception as e:
            raise RuntimeError(
                f"Incomplete ROOT read: file range 0-{n_entries} failed"
            ) from e

        result = {}
        for obj_name, branch_mapping in obj_branches.items():
            available_branches = [
                branch for branch in branch_mapping if branch in file_data.fields
            ]
            if available_branches:
                result[obj_name] = ak.zip({
                    quantity: file_data[full_branch]
                    for full_branch, quantity in branch_mapping.items()
                })

        return result
    
    @staticmethod
    def _auto_detect_branches(tree_branches: set[str]) -> dict[str, dict[str, str]]:
        """
        Auto-detect branch structure when schema is not available.
        Attempts to find branches matching common patterns.
        """
        obj_branches = {}

        object_patterns = {
            "Electrons": ["Electron", "electron", "el"],
            "Muons": ["Muon", "muon", "mu"],
            "Jets": ["Jet", "jet"],
            "Photons": ["Photon", "photon", "gamma"],
            "Taus": ["Tau", "tau", "TauJet", "taujet"]
        }

        required_fields = ["pt", "eta", "phi"]

        for obj_name, patterns in object_patterns.items():
            for pattern in patterns:
                matching_branches = [b for b in tree_branches if pattern.lower() in b.lower()]

                if matching_branches:
                    base_branch = None
                    for branch in matching_branches:
                        parts = branch.split(".")
                        if len(parts) == 2:
                            potential_base = parts[0]
                            has_required = all(
                                f"{potential_base}.{field}" in tree_branches
                                for field in required_fields
                            )
                            if has_required:
                                base_branch = potential_base
                                break

                    if base_branch:
                        available_fields = []
                        for field in required_fields + ["mass"]:
                            if f"{base_branch}.{field}" in tree_branches:
                                available_fields.append(field)

                        if FileParser._can_calculate_inv_mass(available_fields):
                            obj_branches[obj_name] = {
                                f"{base_branch}.{field}": field
                                for field in available_fields
                            }
                        break

        return obj_branches
