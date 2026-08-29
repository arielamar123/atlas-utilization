"""
Base state handler.

Abstract base class for all state handlers.
"""

from abc import ABC, abstractmethod
import logging

from orchestration.context import PipelineContext
from orchestration.states import PipelineState


class StateHandler(ABC):
    """
    Base class for state handlers.
    
    Each state handler implements the logic for transitioning
    from one state to the next.
    """
    
    def __init__(self):
        """Initialize state handler."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        """
        Handle the current state and determine next state.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Tuple of (updated_context, next_state)
            
        Raises:
            Exception: If state handling fails
        """
        pass
    
    def _determine_next_state(self, context: PipelineContext) -> PipelineState:
        """
        Determine next state based on configuration.
        
        Default implementation: follow task order
        (parsing -> mass -> post -> global-range scan -> histogram).
        
        Args:
            context: Current pipeline context
            
        Returns:
            Next pipeline state
        """
        tasks = context.config.tasks
        current = context.current_state

        def histogram_prerequisite() -> PipelineState:
            """Return the next histogram state without redundant scans.

            BumpNet-named SQLite histograms need a global range pass. Single-job
            runs perform it automatically. Batch histogram jobs retain the
            existing contract: one shared ``--scan-only`` job runs before the
            histogram array jobs, so each batch must not repeat the scan.
            """
            hc = context.config.histogram_creation_config
            needs_global_scan = bool(
                hc
                and hc.use_bumpnet_naming
                and context.config.batch_job_index is None
            )
            return (
                PipelineState.GLOBAL_RANGE_SCAN
                if needs_global_scan
                else PipelineState.HISTOGRAM_CREATION
            )
        
        # State transitions based on enabled tasks
        if current == PipelineState.FETCHING_METADATA:
            if tasks.do_parsing:
                return PipelineState.PARSING
            elif tasks.do_mass_calculating:
                return PipelineState.MASS_CALCULATION
            elif tasks.do_post_processing:
                return PipelineState.POST_PROCESSING
            elif tasks.do_histogram_creation:
                return histogram_prerequisite()
            else:
                return PipelineState.COMPLETED
        
        elif current == PipelineState.PARSING:
            if tasks.do_mass_calculating:
                return PipelineState.MASS_CALCULATION
            elif tasks.do_post_processing:
                return PipelineState.POST_PROCESSING
            elif tasks.do_histogram_creation:
                return histogram_prerequisite()
            else:
                return PipelineState.COMPLETED
        
        elif current == PipelineState.MASS_CALCULATION:
            if tasks.do_post_processing:
                return PipelineState.POST_PROCESSING
            elif tasks.do_histogram_creation:
                return histogram_prerequisite()
            else:
                return PipelineState.COMPLETED
        
        elif current == PipelineState.POST_PROCESSING:
            if tasks.do_histogram_creation:
                return histogram_prerequisite()
            else:
                return PipelineState.COMPLETED

        elif current == PipelineState.GLOBAL_RANGE_SCAN:
            if tasks.do_histogram_creation:
                return PipelineState.HISTOGRAM_CREATION
            return PipelineState.COMPLETED
        
        elif current == PipelineState.HISTOGRAM_CREATION:
            return PipelineState.COMPLETED
        
        # Default: COMPLETED
        return PipelineState.COMPLETED
    
    def _log_state_entry(self, context: PipelineContext):
        """Log entry to state."""
        self.logger.info(f"Entering state: {context.current_state}")
    
    def _log_state_exit(self, context: PipelineContext, next_state: PipelineState):
        """Log exit from state."""
        self.logger.info(
            f"Exiting state: {context.current_state} → {next_state}"
        )
