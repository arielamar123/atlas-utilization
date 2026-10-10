"""
FileParser service - Single responsibility: Parse ROOT files.

Extracts events from ATLAS ROOT files using uproot.
No orchestration logic, no state management.
"""

import logging
import json
import awkward as ak
import numpy as np
import uproot
import itertools
from typing import Optional

from services.parsing import schemas
from services.parsing.root_io import open_root_file
from services.parsing.data_run_filter import collision_years_for_file
from services import consts


class PartialFileReadError(RuntimeError):
    """A ROOT file failed after a usable prefix of its events was parsed."""

    def __init__(self, file_path: str, events: ak.Array, read_error: Exception):
        self.file_path = file_path
        self.events = events
        self.read_error = read_error
        super().__init__(
            f"ROOT read failed for {file_path}: {read_error}; "
            f"retaining {len(events)} events parsed before the failure"
        )


class InvalidFileContentError(ValueError):
    """The file was read, but its content cannot be parsed; retrying cannot help."""


def is_transient_read_error(error: BaseException) -> bool:
    """True for network/IO failures that a retry may fix (not a missing file)."""
    if isinstance(error, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return False
    return isinstance(error, (OSError, EOFError))


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
        enable_trigger_matching: bool = False,
        parse_mc: bool = False,
    ) -> Optional[ak.Array]:
        """
        Parse a single ROOT file and return events.
        
        Args:
            file_path: Path or URI to ROOT file
            tree_names: List of possible tree names to search for
            release_year: Release year identifier (e.g., "2024r-pp")
            batch_size: Number of entries to process per batch
            
        Returns:
            Awkward array of events with particle objects, or None after a
            network/read failure (the caller may retry).

        Raises:
            InvalidFileContentError: the file's content cannot be parsed;
                retrying cannot help.
        """
        try:
            with open_root_file(file_path) as root_file:
                return FileParser._parse_opened_file(
                    root_file,
                    tree_names,
                    release_year,
                    batch_size,
                    file_path,
                    enable_jet_tagging,
                    jet_btagging_thresholds,
                    enable_trigger_matching,
                    parse_mc,
                )
        except (PartialFileReadError, InvalidFileContentError):
            raise
        except Exception as e:
            if is_transient_read_error(e):
                logging.warning(f"Failed to read file {file_path}: {e}")
                return None
            raise InvalidFileContentError(
                f"Cannot parse {file_path}: {type(e).__name__}: {e}"
            ) from e
    
    @staticmethod
    def _parse_opened_file(
        root_file,
        tree_names: list[str],
        release_year: str,
        batch_size: int,
        file_path: str,
        enable_jet_tagging: bool,
        jet_btagging_thresholds: Optional[dict[str, float]],
        enable_trigger_matching: bool,
        parse_mc: bool,
    ) -> Optional[ak.Array]:
        """Parse an already-opened ROOT file."""
        tree_name = FileParser._get_data_tree_name(root_file.keys(), tree_names)
        tree = root_file[tree_name]
        all_tree_branches = set(tree.keys())
        n_entries = tree.num_entries
        
        obj_branches = FileParser._extract_branches_by_schema(
            all_tree_branches,
            release_year,
            include_direct_objects=enable_jet_tagging,
            enable_trigger_matching=enable_trigger_matching,
            parse_mc=parse_mc,
        )

        if not obj_branches:
            raise InvalidFileContentError(f"No particles found in schema for file {file_path}")
        if enable_trigger_matching and not parse_mc:
            FileParser._restrict_collision_match_branches(obj_branches, file_path)

        obj_branches = FileParser._filter_accessible_branches(tree, obj_branches)

        if not obj_branches:
            raise InvalidFileContentError(f"No accessible particles found in file {file_path}")
        if enable_trigger_matching and not parse_mc:
            FileParser._require_collision_trigger_branches(obj_branches, file_path)

        collision_match_branches = None
        if enable_trigger_matching and not parse_mc and "_triggerMatch" in obj_branches:
            # Read separately as raw bytes (see _read_collision_matches).
            collision_match_branches = list(obj_branches.pop("_triggerMatch"))
        all_branches = set(itertools.chain.from_iterable(obj_branches.values()))
        batch_reducers = {}
        if enable_trigger_matching and not parse_mc:
            # Collision data: reduce trigger branches to per-event booleans
            # batch by batch instead of keeping the raw bitsets and links.
            if "_triggerDecisionRaw" in obj_branches:
                batch_reducers["_triggerDecisionRaw"] = (
                    "_triggerDecision",
                    FileParser._collision_trigger_reducer(root_file, file_path),
                )
        obj_events, read_error = FileParser._read_file_in_batches(
            tree,
            all_branches,
            obj_branches,
            n_entries,
            batch_size,
            batch_reducers,
        )
        if collision_match_branches and "_dataRunNumber" in obj_events:
            obj_events["_triggerMatch"] = FileParser._read_collision_matches(
                tree, collision_match_branches, len(obj_events["_dataRunNumber"]), file_path
            )
        obj_events = FileParser._split_combined_leptons(obj_events, release_year)
        if enable_jet_tagging:
            obj_events = FileParser._calculate_btagging_and_split(obj_events, jet_btagging_thresholds)
        # Strip out DirectObjects -- they are not physics objects!
        if "DirectObjects" in obj_events.keys():
            obj_events.pop("DirectObjects")
        events = ak.zip(obj_events, depth_limit=1)
        if read_error is not None:
            raise PartialFileReadError(file_path, events, read_error) from read_error
        return events

    @staticmethod
    def _split_combined_leptons(
        obj_events: dict[str, ak.Array], release_year: str
    ) -> dict[str, ak.Array]:
        """Split legacy ``lep_*`` branches using their absolute PDG identifier."""
        normalized_release = schemas.normalize_release_year(release_year)
        if normalized_release not in {"2016e-8tev", "2025e-13tev-beta"}:
            return obj_events

        source = obj_events.get("Electrons")
        if source is None:
            source = obj_events.get("Muons")
        # A restricted allow-list may intentionally omit both combined-lepton
        # collections, in which case there is nothing to split.
        if source is None:
            return obj_events
        if "type" not in source.fields:
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
        if "Jets" not in obj_events or "DirectObjects" not in obj_events:
            # No jets to tag. Move on.
            return obj_events

        if "Jet_btagDeepFlavB" in obj_events["DirectObjects"].fields:
            # CMS: Discriminant is pre-calculated as the Jet_btagDeepFlavB field. Can change to a different algorithm if needed.
            # See https://cms-opendata-workshop.github.io/workshop2024-lesson-physics-objects/instructor/05-btagging.html
            is_bjet = obj_events["DirectObjects"]["Jet_btagDeepFlavB"] > jet_btagging_thresholds["Jet_btagDeepFlavB"]
        elif "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb" in obj_events["DirectObjects"].fields:
            # ATLAS
            indices = obj_events["DirectObjects"]["AnalysisJetsAuxDyn.btaggingLink/AnalysisJetsAuxDyn.btaggingLink.m_persIndex"]
            pb = obj_events["DirectObjects"]["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb"][indices]
            pc = obj_events["DirectObjects"]["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc"][indices]
            pu = obj_events["DirectObjects"]["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu"][indices]
            # DL1d score: log(pb / (fc*pc + (1-fc)*pu)), fc=0.018 is standard ATLAS.
            fc = 0.018

            # Handle edge cases where pb=0, or pc=pu=0.
            denominator = fc * pc + (1 - fc) * pu
            is_scoreable = (pb > 0) & (denominator > 0)
            dl1d = np.log(
                ak.where(is_scoreable, pb, 1.0) / ak.where(is_scoreable, denominator, 1.0)
            )
            # For now, if not scoreable -- assume jet.
            is_bjet = ak.where(
                is_scoreable,
                dl1d > jet_btagging_thresholds["DL1d"],
                False,
            )
        else:
            return obj_events
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
        release_year: str,
        include_direct_objects: bool = False,
        enable_trigger_matching: bool = False,
        parse_mc: bool = False,
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
        if include_direct_objects:
            obj_branches.update({"DirectObjects": {k: k for k in direct_objects}})

        # Trigger matching branches (event-level, per-particle ElementLink vectors).
        # Each branch is a ``var * var * ElementLink``; a non-empty inner list means
        # that offline particle matched the HLT trigger object within ΔR < 0.07.
        # Stored under ``_triggerMatch`` so downstream code can distinguish them
        # from particle-type fields.
        if enable_trigger_matching:
            trigger_branches = schemas.get_all_trigger_branches()
            if not parse_mc:
                # Collision data has the same match decorations as MC; read
                # the ones of the collision-data menu.  A missing branch means
                # no lepton matched that chain anywhere in the file.
                trigger_branches = schemas.get_data_trigger_match_branches()
            available_trigger = [b for b in trigger_branches if b in tree_branches]
            if available_trigger:
                obj_branches["_triggerMatch"] = {b: b for b in available_trigger}
            # Do not infer the data/MC mode from branch availability.  The
            # pipeline configuration is authoritative and collision data can
            # contain both run-number decorations.
            if parse_mc and schemas.RANDOM_RUN_NUMBER_BRANCH in tree_branches:
                obj_branches["_runNumber"] = {schemas.RANDOM_RUN_NUMBER_BRANCH: "_runNumber"}
            elif not parse_mc and schemas.DATA_RUN_NUMBER_BRANCH in tree_branches:
                obj_branches["_dataRunNumber"] = {schemas.DATA_RUN_NUMBER_BRANCH: "_dataRunNumber"}
            if not parse_mc:
                decision_branches = {
                    schemas.TRIGGER_DECISION_SMK_BRANCH: "smk",
                    schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH: "hlt_passed_physics",
                }
                # Missing branches are reported by _require_collision_trigger_branches.
                if all(b in tree_branches for b in decision_branches):
                    obj_branches["_triggerDecisionRaw"] = decision_branches

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
        except Exception as error:
            # A network failure must be retried, not mistaken for an
            # unreadable branch.
            if is_transient_read_error(error):
                raise
            for branch_path in all_candidate_branches:
                try:
                    test_arr = tree.arrays(
                        branch_path,
                        entry_start=0, entry_stop=1,
                        library="ak"
                    )
                    if branch_path in test_arr.fields:
                        accessible_set.add(branch_path)
                except Exception as branch_error:
                    if is_transient_read_error(branch_error):
                        raise
                    continue
        
        accessible_obj_branches = {}
        for obj_name, branch_mapping in obj_branches.items():
            accessible_branches = {
                bp: qty for bp, qty in branch_mapping.items()
                if bp in accessible_set
            }
            if obj_name in ("DirectObjects", "_triggerMatch", "_runNumber", "_dataRunNumber", "_triggerDecisionRaw"):
                # These are not particle types — skip the inv-mass field check.
                if accessible_branches:
                    accessible_obj_branches[obj_name] = accessible_branches
            elif accessible_branches and FileParser._can_calculate_inv_mass(
                list(accessible_branches.values())
            ):
                accessible_obj_branches[obj_name] = accessible_branches
        
        return accessible_obj_branches
    
    @staticmethod
    def _read_file_in_batches(
        tree,
        all_branches: set[str],
        obj_branches: dict[str, dict[str, str]],
        n_entries: int,
        batch_size: int,
        batch_reducers: Optional[dict] = None,
    ) -> tuple[dict[str, ak.Array], Optional[Exception]]:
        # batch_reducers: {obj_name: (output_name, fn)} reduces that object's
        # branches batch by batch; the concatenated result is stored as-is.
        batch_reducers = batch_reducers or {}
        obj_events_by_quantities = {
            obj_name: [] for obj_name in obj_branches.keys()
        }
        read_error = None
        
        is_file_big = n_entries > batch_size
        if is_file_big:
            entry_ranges = [
                (start, min(start + batch_size, n_entries))
                for start in range(0, n_entries, batch_size)
            ]
        else:
            entry_ranges = [(0, n_entries)]
        
        for entry_start, entry_stop in entry_ranges:
            try:
                batch_data = tree.arrays(
                    all_branches,
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    library="ak"
                )
            except Exception as e:
                read_error = RuntimeError(
                    f"batch {entry_start}-{entry_stop} failed with "
                    f"{type(e).__name__}: {e}"
                )
                logging.warning("Stopping partial ROOT read: %s", read_error)
                break
            
            for obj_name, branch_mapping in obj_branches.items():
                available_branches = [
                    b for b in branch_mapping.keys() if b in batch_data.fields
                ]
                if available_branches:
                    subset = batch_data[available_branches]
                    if len(subset) > 0:
                        if obj_name in batch_reducers:
                            subset = batch_reducers[obj_name][1](subset)
                        obj_events_by_quantities[obj_name].append(subset)

        result = {}
        for obj_name, chunks in obj_events_by_quantities.items():
            if chunks:
                concatenated = ak.concatenate(chunks)
                if obj_name in batch_reducers:
                    result[batch_reducers[obj_name][0]] = concatenated
                elif obj_name == "_triggerMatch":
                    # Trigger branches are ``var * var * ElementLink``.
                    # We collapse each to a single per-event boolean:
                    # True if ANY particle in the event has a non-empty match
                    # for that chain.  The result is a record of booleans keyed
                    # by the original branch name.
                    trig_fields = {}
                    for full_branch in obj_branches[obj_name].keys():
                        if full_branch not in concatenated.fields:
                            continue
                        raw = concatenated[full_branch]
                        # raw[i] holds one entry per matched combination in event i;
                        # raw[i][j] links the offline particle(s) of combination j.
                        # The event matched if any combination is non-empty.
                        per_particle_matched = ak.num(raw, axis=2) > 0
                        trig_fields[full_branch] = ak.any(per_particle_matched, axis=1)
                    if trig_fields:
                        result[obj_name] = ak.zip(trig_fields)
                elif obj_name == "_runNumber":
                    result[obj_name] = concatenated[schemas.RANDOM_RUN_NUMBER_BRANCH]
                elif obj_name == "_dataRunNumber":
                    result[obj_name] = concatenated[schemas.DATA_RUN_NUMBER_BRANCH]
                else:
                    result[obj_name] = ak.zip({
                        quantity: concatenated[full_branch]
                        for full_branch, quantity in obj_branches[obj_name].items()
                    })
        
        return result, read_error

    @staticmethod
    def _restrict_collision_match_branches(obj_branches: dict, file_path: str) -> None:
        """Keep only the match branches of the chains for this file's run year.

        Every collision-data dataset holds one run, so a file needs only its
        year's chains.  Reading these nested ElementLink branches dominates the
        trigger cost.  Files with an unknown run keep every chain.
        """
        years = collision_years_for_file(file_path)
        if years is None or "_triggerMatch" not in obj_branches:
            return
        wanted = {
            schemas.data_trigger_match_branch(chain)
            for year in years
            for chains in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS[year].values()
            for chain in chains
        }
        kept = {b: q for b, q in obj_branches["_triggerMatch"].items() if b in wanted}
        if kept:
            obj_branches["_triggerMatch"] = kept
        else:
            obj_branches.pop("_triggerMatch")

    @staticmethod
    def _read_collision_matches(tree, branches: list[str], n_entries: int, file_path: str) -> ak.Array:
        """Per event and chain: True if any trigger-matched combination is non-empty.

        Same result as the MC path's ``any(num(raw, axis=2) > 0)``, but the
        branches are read as raw entry bytes instead of being deserialized
        into nested ElementLink records, which dominated the trigger cost.
        """
        raw_bytes = uproot.interpretation.jagged.AsJagged(uproot.AsDtype("u1"), header_bytes=0)
        fields = {}
        for branch in branches:
            raw = tree[branch].array(interpretation=raw_bytes, entry_stop=n_entries, library="ak")
            matched = FileParser.trigger_matches_from_raw_entries(raw)
            if matched is None:
                logging.warning(
                    "Unexpected TrigMatchedObjects byte layout in %s for %s; "
                    "deserializing the branch instead", file_path, branch,
                )
                objects = tree[branch].array(entry_stop=n_entries, library="ak")
                matched = ak.to_numpy(ak.any(ak.num(objects, axis=2) > 0, axis=1))
            fields[branch] = matched
        return ak.zip({branch: ak.Array(values) for branch, values in fields.items()})

    @staticmethod
    def trigger_matches_from_raw_entries(raw: ak.Array) -> Optional[np.ndarray]:
        """Decode ``vector<vector<ElementLink>>`` entries to "any non-empty inner vector".

        Entry layout: 6-byte header (byte count, version), outer size (uint32,
        big-endian), then per inner vector its size (uint32) followed by that
        many ElementLink objects, each starting with a ROOT byte-count word.
        Returns None if any entry does not follow this layout.
        """
        lengths = ak.to_numpy(ak.num(raw)).astype(np.int64)
        if not len(lengths):
            return np.zeros(0, dtype=bool)
        flat = ak.to_numpy(ak.flatten(raw)).astype(np.uint8, copy=False)
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
        if np.any(lengths < 10):
            return None
        size_bytes = flat[starts[:, None] + np.arange(6, 10)].astype(np.uint64)
        outer = (size_bytes * np.array([1 << 24, 1 << 16, 1 << 8, 1], dtype=np.uint64)).sum(axis=1)
        if np.any(lengths[outer == 0] != 10):
            return None
        matched = np.zeros(len(lengths), dtype=bool)
        for entry in np.nonzero(outer)[0]:
            pos, end = int(starts[entry]) + 10, int(starts[entry] + lengths[entry])
            for _ in range(int(outer[entry])):
                if pos + 4 > end:
                    return None
                count = int.from_bytes(flat[pos:pos + 4].tobytes(), "big")
                pos += 4
                matched[entry] |= count > 0
                for _ in range(count):
                    if pos + 4 > end:
                        return None
                    pos += (int.from_bytes(flat[pos:pos + 4].tobytes(), "big") & 0x3FFFFFFF) + 4
            if pos != end:
                return None
        return matched

    @staticmethod
    def _require_collision_trigger_branches(obj_branches: dict, file_path: str) -> None:
        """Reject a collision-data file whose trigger selection cannot be evaluated."""
        readable = {
            branch
            for name in ("_triggerDecisionRaw", "_dataRunNumber")
            for branch in obj_branches.get(name, {})
        }
        missing = [
            branch for branch in (
                schemas.TRIGGER_DECISION_SMK_BRANCH,
                schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH,
                schemas.DATA_RUN_NUMBER_BRANCH,
            )
            if branch not in readable
        ]
        if missing:
            raise InvalidFileContentError(
                f"Collision-data file {file_path} lacks readable trigger branches "
                f"{missing}; its events cannot be trigger-selected"
            )

    @staticmethod
    def _collision_trigger_reducer(root_file, file_path: str):
        """Return a per-batch decoder of xTrigDecision HLT physics bits.

        The 256-word bitsets are decoded batch by batch so that only the
        configured chains' booleans are kept in memory.  This is deliberately
        collision-data-only; MC uses its AnalysisTrigMatch parsing path.
        """
        menus = FileParser._read_hlt_menus(root_file, file_path)
        configured = sorted({
            chain
            for by_object in schemas.DATA_SINGLE_LEPTON_TRIGGER_CHAINS.values()
            for chains in by_object.values()
            for chain in chains
        })
        reported: set[int] = set()

        def decode(batch: ak.Array) -> ak.Array:
            missing = [
                branch for branch in (
                    schemas.TRIGGER_DECISION_SMK_BRANCH,
                    schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH,
                )
                if branch not in batch.fields
            ]
            if missing:
                raise InvalidFileContentError(
                    f"Collision-data file {file_path} lacks readable HLT decision branches {missing}"
                )
            return FileParser.decode_hlt_physics_decisions(
                batch[schemas.TRIGGER_DECISION_SMK_BRANCH],
                batch[schemas.TRIGGER_DECISION_HLT_PHYSICS_BRANCH],
                menus, configured, file_path, reported,
            )

        return decode

    @staticmethod
    def _read_hlt_menus(root_file, file_path: str) -> dict:
        """Read every TriggerMenuJson_HLT entry, keyed by SMK."""
        try:
            metadata = root_file["MetaData"]
            menu = metadata.arrays(
                [schemas.TRIGGER_MENU_KEY_BRANCH, schemas.TRIGGER_MENU_PAYLOAD_BRANCH],
                library="ak",
            )
            # A merged file can hold one MetaData entry per input file.
            menus = {
                int(key): json.loads(payload)
                for keys, payloads in zip(
                    ak.to_list(menu[schemas.TRIGGER_MENU_KEY_BRANCH]),
                    ak.to_list(menu[schemas.TRIGGER_MENU_PAYLOAD_BRANCH]),
                )
                for key, payload in zip(keys, payloads)
            }
        except Exception as error:
            if is_transient_read_error(error):
                raise
            raise InvalidFileContentError(
                f"Could not read TriggerMenuJson_HLT metadata in collision-data file "
                f"{file_path}: {type(error).__name__}: {error}"
            ) from error
        if not menus:
            raise InvalidFileContentError(
                f"Collision-data file {file_path} has no TriggerMenuJson_HLT menus"
            )
        return menus

    @staticmethod
    def decode_hlt_physics_decisions(
        smk, passed_words, menus: dict, chains: list[str], file_path: str = "",
        reported: Optional[set] = None,
    ) -> ak.Array:
        """Return one boolean field per chain from per-event HLT bitsets.

        Bit ``counter`` of ``passed_words`` (word ``counter // 32``) is the
        chain's decision, where ``counter`` comes from the event's own SMK
        menu.  A chain absent from that menu was not run, so it did not pass.
        An SMK with no menu in the file is an error: its bits cannot be
        interpreted.
        """
        smk = ak.to_numpy(smk).astype(np.int64)
        decoded = {chain: np.zeros(len(smk), dtype=bool) for chain in chains}
        for key in np.unique(smk):
            if int(key) not in menus:
                raise InvalidFileContentError(
                    f"No TriggerMenuJson_HLT entry for SMK {int(key)} in "
                    f"collision-data file {file_path}; available: {sorted(menus)}"
                )
            menu_chains = menus[int(key)].get("chains", {})
            selected = smk == key
            words = passed_words[selected]
            missing = []
            for chain in chains:
                entry = menu_chains.get(chain)
                if entry is None:
                    missing.append(chain)
                    continue
                word_index, bit = divmod(int(entry["counter"]), 32)
                padded = ak.pad_none(words, word_index + 1, axis=1)
                word = ak.to_numpy(ak.fill_none(padded[:, word_index], 0)).astype(np.uint64)
                decoded[chain][selected] = ((word >> bit) & 1).astype(bool)
            if missing and (reported is None or int(key) not in reported):
                if reported is not None:
                    reported.add(int(key))
                logging.info(
                    "Trigger menu SMK %d in %s does not contain chains %s; "
                    "they did not run for its events",
                    int(key), file_path, missing,
                )
        return ak.zip({chain: ak.Array(values) for chain, values in decoded.items()})
    
    @staticmethod
    def _auto_detect_branches(
        tree_branches: set[str]
    ) -> dict[str, dict[str, str]]:
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
