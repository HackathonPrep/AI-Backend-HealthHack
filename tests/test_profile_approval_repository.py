from types import SimpleNamespace

from postgrest.exceptions import APIError

from app.schemas.profile import ProfileApprovalRequest
from app.services.records import RecordRepository, _transient_profile_approvals


class FakeQuery:
    def __init__(self, client: "FakeClient", table: str) -> None:
        self.client = client
        self.table_name = table
        self.operation = "select"
        self.payload = None

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args):
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def execute(self):
        if (
            self.table_name == "participant_profile_approvals"
            and self.client.missing_approvals_table
        ):
            raise APIError(
                {
                    "message": "Could not find the table in the schema cache",
                    "code": "PGRST205",
                    "hint": None,
                    "details": None,
                }
            )
        if self.table_name == "patients":
            return SimpleNamespace(data=[{"id": "patient-1"}])
        if self.table_name == "service_catalog":
            return SimpleNamespace(
                data=[{"id": "support-worker", "name": "Support worker", "active": True}]
            )
        if self.table_name == "participant_profile_approvals" and self.operation == "insert":
            self.client.inserted_approval = self.payload
            return SimpleNamespace(
                data=[
                    {
                        "id": "approval-1",
                        "created_at": "2026-07-19T00:00:00+00:00",
                        **self.payload,
                    }
                ]
            )
        if self.table_name == "participant_profile_approvals":
            return SimpleNamespace(data=self.client.approvals)
        return SimpleNamespace(data=[])


class FakeClient:
    def __init__(self) -> None:
        self.inserted_approval = None
        self.approvals = []
        self.missing_approvals_table = False

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)


def approval_request() -> ProfileApprovalRequest:
    return ProfileApprovalRequest.model_validate(
        {
            "sections": [
                {
                    "id": "supports",
                    "title": "Recommended supports",
                    "status": "confirmed",
                    "items": ["Support worker — high priority"],
                }
            ],
            "consents": ["reviewed", "matching", "no_automatic_document_sharing"],
        }
    )


def test_save_profile_approval_appends_a_snapshot_without_mutating_source_records() -> None:
    repository = object.__new__(RecordRepository)
    repository.client = FakeClient()
    repository.configured_demo_patient_id = "patient-1"

    result = repository.save_profile_approval(approval_request())

    assert result["id"] == "approval-1"
    assert repository.client.inserted_approval["patient_id"] == "patient-1"
    assert repository.client.inserted_approval["approved_profile"]["support_catalog_item_ids"] == [
        "support-worker"
    ]
    assert repository.client.inserted_approval["approved_profile"]["sections"][0]["status"] == "confirmed"


def test_latest_profile_approval_returns_the_newest_snapshot() -> None:
    repository = object.__new__(RecordRepository)
    repository.client = FakeClient()
    repository.configured_demo_patient_id = "patient-1"
    repository.client.approvals = [{"id": "latest", "approved_profile": {}}]

    assert repository.latest_profile_approval() == {"id": "latest", "approved_profile": {}}


def test_profile_approval_falls_back_when_the_migration_has_not_been_applied() -> None:
    _transient_profile_approvals.clear()
    repository = object.__new__(RecordRepository)
    repository.client = FakeClient()
    repository.configured_demo_patient_id = "patient-1"
    repository.client.missing_approvals_table = True

    approval = repository.save_profile_approval(approval_request())

    assert approval["id"].startswith("transient-")
    assert approval["approved_profile"]["sections"][0]["id"] == "supports"
    assert repository.latest_profile_approval() == approval
