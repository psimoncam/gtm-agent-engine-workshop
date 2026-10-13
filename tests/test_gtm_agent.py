import os
import unittest
from unittest.mock import patch
from uuid import UUID

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from gtm_agent import gtm_agent as gtm_agent_module


GTM_TOOLS = [
    gtm_agent_module.lookup_offering,
    gtm_agent_module.build_prospect_profile,
    gtm_agent_module.get_prospect,
    gtm_agent_module.send_prospect_email,
    gtm_agent_module.score_prospect,
    gtm_agent_module.update_prospect_info,
    gtm_agent_module.get_current_rep,
]


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class GTMAgentTests(unittest.TestCase):
    def test_production_agent_contains_only_gtm_tools(self):
        tool_node = gtm_agent_module.gtm_agent.nodes["tools"].bound

        self.assertEqual(
            set(tool_node.tools_by_name),
            {tool.name for tool in GTM_TOOLS},
        )
        self.assertEqual(set(gtm_agent_module.gtm_agent.nodes), {"__start__", "model", "tools"})

    def test_agent_calls_gtm_tools_and_returns_response(self):
        model = ScriptedChatModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep",
                "args": {},
                "id": "rep-call",
                "type": "tool_call",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "lookup_offering",
                "args": {"offering_id": "OFFER-10001"},
                "id": "offering-call",
                "type": "tool_call",
            }]),
            AIMessage(content="Ava Mills supports Core Platform Suite."),
        ])
        agent = create_agent(model, tools=GTM_TOOLS, system_prompt=gtm_agent_module.SYSTEM_PROMPT)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Who is my rep and what is OFFER-10001?"}]},
            config={"metadata": {"user_id": "rep_amills"}},
        )

        tool_calls = [
            call["name"]
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(tool_calls, ["get_current_rep", "lookup_offering"])
        self.assertIn("Ava Mills", tool_messages[0].content)
        self.assertIn("Core Platform Suite", tool_messages[1].content)
        self.assertEqual(result["messages"][-1].content, "Ava Mills supports Core Platform Suite.")

    def test_send_email_uses_runtime_rep_context(self):
        model = ScriptedChatModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"name": "Sam Lee", "email": "sam@example.com"},
                    "subject": "Following up",
                    "body": "Thanks for speaking with me.",
                },
                "id": "email-call",
                "type": "tool_call",
            }]),
            AIMessage(content="Email sent from Ava Mills."),
        ])
        agent = create_agent(model, tools=GTM_TOOLS, system_prompt=gtm_agent_module.SYSTEM_PROMPT)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Email Sam Lee"}]},
            config={"metadata": {"user_id": "rep_amills"}},
        )

        email_result = result["messages"][-2]
        self.assertIn("Ava Mills", email_result.content)
        self.assertIn("ava.mills@northpoint.com", email_result.content)
        self.assertEqual(result["messages"][-1].content, "Email sent from Ava Mills.")

    def test_run_agent_preserves_run_configuration(self):
        class CapturingAgent:
            def invoke(self, inputs, config):
                self.inputs = inputs
                self.config = config
                return {"messages": [AIMessage(content="Done.")]}

        capturing_agent = CapturingAgent()
        with patch.object(gtm_agent_module, "gtm_agent", capturing_agent):
            result = gtm_agent_module.run_agent(
                "Email Sam about the update",
                user_id="rep_amills",
                environment="test",
                thread_id="thread-123",
            )

        config = capturing_agent.config
        self.assertEqual(config["run_name"], "GTM Assistant")
        self.assertIsInstance(config["run_id"], UUID)
        self.assertEqual(config["metadata"], {
            "thread_id": "thread-123",
            "user_id": "rep_amills",
            "environment": "test",
            "request_intent": "send_email",
        })
        self.assertEqual(result["reply"], "Done.")
        self.assertEqual(result["run_id"], str(config["run_id"]))


if __name__ == "__main__":
    unittest.main()
