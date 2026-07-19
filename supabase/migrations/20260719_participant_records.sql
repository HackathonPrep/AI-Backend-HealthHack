create table if not exists public.participant_records (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  record_type text not null check (record_type in ('document_plan', 'action_pack')),
  source_filename text,
  clinical_extraction jsonb not null,
  result jsonb not null,
  created_at timestamptz not null default now()
);
alter table public.participant_records enable row level security;
create policy "owners manage their participant records" on public.participant_records
for all to authenticated using (owner_id = auth.uid()) with check (owner_id = auth.uid());
