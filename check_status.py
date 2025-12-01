import requests
from inference_auth_token import get_access_token

from datetime import datetime

def dedup_models(models):
    """
    models: list of (model, est) tuples
    Removes duplicates, keeping the *earlier* est for each model.
    Also strips '2025 (Chicago time)' from the est string.
    """
    cleaned = {}
    fmt = "%a %b %d %H:%M:%S %Y"  # strptime format without '(Chicago time)'

    for model, est in models:
        # Strip the trailing '2025 (Chicago time)'
        est_stripped = est.replace("2025 (Chicago time)", "").strip()

        try:
            est_dt = datetime.strptime(est_stripped, fmt)
        except ValueError:
            # if parsing fails, keep raw string
            est_dt = None

        if model not in cleaned:
            cleaned[model] = (est_dt, est_stripped)
        else:
            # keep the earlier one
            prev_dt, prev_str = cleaned[model]
            if est_dt and prev_dt:
                if est_dt < prev_dt:
                    cleaned[model] = (est_dt, est_stripped)
            elif est_dt:  # if previous wasn't parsable, prefer valid datetime
                cleaned[model] = (est_dt, est_stripped)

    # return as list of (model, cleaned_est)
    return [(m, est_str) for m, (_, est_str) in cleaned.items()]

def retrieve_model_list(j, key):
    l = j[key]
    #for m in l:
    #    model = m['Models']
    #    print('MM', model, m)
    models = [(m['Models'], m.get('Estimated Start Time', '')) for m in l]
    all_models = [(name.strip(), est.replace('  ',' ')) for (s,est) in models for name in s.split(",")]
    #print(f'\n{key}: {all_models}\n')
    all_models = sorted(list(set(all_models)))
    return all_models

access_token = get_access_token()

url = "https://inference-api.alcf.anl.gov/resource_server/sophia/jobs"

headers = { "Authorization": f"Bearer {access_token}" }

response = requests.get(url, headers=headers, timeout=30)

if not response.ok:
    print(f"Error {response.status_code}: {response.text}")
    exit(1)

j = response.json()

running   = retrieve_model_list(j, 'running')
queued    = retrieve_model_list(j, 'queued')
# Dedup because it seems that the same model can be queued more than once
scheduled = dedup_models([(model, est) for (model, est) in queued if est != ''])
# List a model as "other" only if not already queued, which apparently it can be
others    = [model for (model, est) in queued if est == '' and model not in [m for (m, _) in scheduled]]

print(f'{len(running)} running models:')
for (model,_) in running:
    print(f'    {model}')

print(f'{len(scheduled)} queued models, with estimated start times:')
for (model, est) in scheduled:
    print(f'    {model}: {est}')

print(f'{len(others)} non-queued models:')
for model in others:
    print(f'    {model}')

print(f'Total {len(running)+len(scheduled)+len(others)} models: {len(running)} running, {len(scheduled)} queued, {len(others)} not queued.')
