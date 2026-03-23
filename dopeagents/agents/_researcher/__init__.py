"""Agent-private infrastructure for DeepResearcher."""

from dopeagents.agents._researcher.claim_analysis import Claim, ClaimCluster, CrossReferenceOutput
from dopeagents.agents._researcher.confidence import ConfidenceBreakdown, ConfidenceCalculator
from dopeagents.agents._researcher.hybrid_step import HybridStepResult, HybridStepRunner
from dopeagents.agents._researcher.memory import ResearchMemory, ResearchSession
from dopeagents.agents._researcher.model_capability import ModelCapability, detect_capability
from dopeagents.agents._researcher.progress import (
    ResearchProgress,
    StepProgress,
    StepStatus,
    console_progress_callback,
)
from dopeagents.agents._researcher.report_generator import (
    Citation,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    StructuredReport,
)
from dopeagents.agents._researcher.tools import (
    ANALYSIS_TOOLS,
    ToolBudget,
    ToolCall,
    ToolExecutor,
    ToolName,
    ToolResult,
)

__all__ = [
    "ANALYSIS_TOOLS",
    "Citation",
    "Claim",
    "ClaimCluster",
    "ConfidenceBreakdown",
    "ConfidenceCalculator",
    "CrossReferenceOutput",
    "HybridStepResult",
    "HybridStepRunner",
    "ModelCapability",
    "ReportFormat",
    "ReportGenerator",
    "ReportSection",
    "ResearchMemory",
    "ResearchProgress",
    "ResearchSession",
    "StepProgress",
    "StepStatus",
    "StructuredReport",
    "ToolBudget",
    "ToolCall",
    "ToolExecutor",
    "ToolName",
    "ToolResult",
    "console_progress_callback",
    "detect_capability",
]
