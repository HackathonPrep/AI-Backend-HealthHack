-- Immutable participant-confirmed snapshots. Source documents and AI extractions
-- remain untouched so their provenance can be reviewed independently.
create table if not exists public.participant_profile_approvals (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  approved_profile jsonb not null,
  follow_up_answers jsonb not null default '[]'::jsonb,
  consents jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists participant_profile_approvals_patient_created_idx
  on public.participant_profile_approvals (patient_id, created_at desc);
