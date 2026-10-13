import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_records import PROSPECTS
from gtm_agent.gtm_agent import build_prospect_profile, score_prospect

os.environ["LANGSMITH_TRACING"] = "false"


class ProspectUpdateTests(unittest.TestCase):
    prospect_id = "LEAD-71001"

    def setUp(self):
        self.original_tech_stack = list(PROSPECTS[self.prospect_id]["tech_stack"])
        self.original_profile = data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        PROSPECTS[self.prospect_id]["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.pop(self.prospect_id, None)
        if self.original_profile is not None:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Kafka", result["tech_stack"])
        self.assertIn("Kafka", PROSPECTS[self.prospect_id]["tech_stack"])

    def test_update_invalidates_cached_profile(self):
        initial_profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        self.assertNotIn("Kafka", initial_profile["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        rebuilt_profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]

        self.assertIn("Kafka", rebuilt_profile["tech_stack"])

    @patch("gtm_agent.gtm_agent._scoring_llm")
    def test_rescore_receives_updated_technology(self, scoring_llm):
        profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        data_service.update_prospect_info(self.prospect_id, "Kafka")

        def score_from_request(messages):
            score_request = messages[1]["content"]
            scored_profile = json.loads(
                score_request.split("\n\nProspect profile:\n", 1)[1]
            )
            technology_present = "Kafka" in scored_profile["tech_stack"]
            return Mock(
                model_dump=Mock(
                    return_value={
                        "score": 90,
                        "max_score": 100,
                        "justification": (
                            "Kafka is present in the prospect's technology stack."
                            if technology_present
                            else "Kafka is missing from the prospect's technology stack."
                        ),
                        "rubric_breakdown": {
                            "revenue_fit": 30,
                            "tech_stack_match": 30,
                            "segment_fit": 30,
                            "component_max": 100,
                        },
                    }
                )
            )

        scoring_llm.invoke.side_effect = score_from_request

        result = score_prospect.invoke(
            {
                "prospect_profile": profile,
                "offering": data_service.get_offering("OFFER-10005"),
            }
        )

        score_request = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(score_request.split("\n\nProspect profile:\n", 1)[1])
        self.assertIn("Kafka", scored_profile["tech_stack"])
        self.assertNotIn("missing Kafka", result["justification"])

    def test_update_unknown_prospect_remains_not_found(self):
        self.assertEqual(
            data_service.update_prospect_info("UNKNOWN-LEAD", "Kafka"),
            {"updated": False, "found": False},
        )


if __name__ == "__main__":
    unittest.main()
