"""Builder pattern for agent configuration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.models import AgentConfig

from config.models import AgentConfig, PromptConfig, ToolConfig


class AgentConfigBuilder:
    """Fluent interface for building agent configurations."""

    def __init__(self, agent_id: str):
        self._config = AgentConfig(
            agent_id=agent_id,
            prompts=[],
            tools=[],
            metadata={}
        )

    def with_prompt(self, role: str, content: str, **template_vars) -> 'AgentConfigBuilder':
        """Add a prompt configuration."""
        self._config.prompts.append(
            PromptConfig(role=role, content=content, template_vars=template_vars)
        )
        return self

    def with_tool(self, name: str, description: str, **kwargs) -> 'AgentConfigBuilder':
        """Add a tool configuration."""
        self._config.tools.append(
            ToolConfig(name=name, description=description, parameters=kwargs)
        )
        return self

    def with_metadata(self, **metadata) -> 'AgentConfigBuilder':
        """Add metadata to the configuration."""
        self._config.metadata.update(metadata)
        return self

    def build(self) -> AgentConfig:
        """Build and return the final configuration."""
        return self._config
