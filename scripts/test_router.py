from __future__ import annotations
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from backend.agents.router_agent import route_request as route

def main():
    user_id = 'smoketest-user'
    queries = sys.argv[1:] or [
        'plan a cheap high-protein lunch for today',
        'log my squat sets from this morning',
        'swap dinner to a salmon bowl',
        'add 3x5 bench press at 135 lbs',
    ]
    results = []
    for q in queries:
        try:
            agent, payload = route(user_id=user_id, query=q)
            print(f'ROUTED: {q} -> {agent}')
            results.append((q, agent))
        except Exception as e:
            print(f'Router error on: {q} | {e}')
            results.append((q, None))
        time.sleep(0.05)

    print("\nSummary:")
    for q, a in results:
        print(f'- {q} -> {a}')

    try:
        from backend.services.supabase_service import supabase
        rows = (
            supabase.table('agent_decisions')
            .select('*')
            .order('created_at', desc=True)
            .limit(5)
            .execute()
            .data
            or []
        )
        print("\nRecent agent_decisions (latest 5):")
        for r in rows:
            reason = (r.get('extra') or {}).get('reason')
            print(
                f"- {r.get('created_at')} | {r.get('agent_slug')} | success={r.get('was_success')} "
                f"| reason={reason} | q='{(r.get('query_text') or '')[:60]}'"
            )
    except Exception:
        pass

if __name__ == '__main__':
    main()
