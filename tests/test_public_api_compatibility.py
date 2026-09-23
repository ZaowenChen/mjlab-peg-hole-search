import unittest


class PublicApiCompatibilityTests(unittest.TestCase):
    def test_control_aliases_keep_class_identity(self):
        from mjlab_contact_prep.controller import ContactController as Legacy
        from mjlab_contact_prep.control.contact import ContactController as Current

        self.assertIs(Current, Legacy)

    def test_environment_aliases_keep_class_identity(self):
        from mjlab_contact_prep.search_env import SearchEnv as Legacy
        from mjlab_contact_prep.envs.search import SearchEnv as Current

        self.assertIs(Current, Legacy)

    def test_training_and_evaluation_are_separate_public_apis(self):
        from mjlab_contact_prep.evaluation.formal import evaluate_job
        from mjlab_contact_prep.night_run.jobs import evaluate_job as LegacyEvaluate
        from mjlab_contact_prep.night_run.jobs import train_job as LegacyTrain
        from mjlab_contact_prep.training.formal import train_job

        self.assertIs(train_job, LegacyTrain)
        self.assertIs(evaluate_job, LegacyEvaluate)


if __name__ == "__main__":
    unittest.main()
