"""Ambil posting X/Twitter asli memakai API resmi X (perlu akses API)."""
import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

QUERY = '("mitigasi bencana" OR "kesiapsiagaan bencana" OR "peringatan dini" OR "evakuasi bencana") lang:id -is:retweet'
ENDPOINT = 'https://api.x.com/2/tweets/search/recent'


def main():
    parser = argparse.ArgumentParser(description='Unduh posting X terkait mitigasi bencana')
    parser.add_argument('--target', type=int, default=350)
    parser.add_argument('--output', default='tweet_data.csv')
    parser.add_argument('--query', default=QUERY)
    args = parser.parse_args()
    load_dotenv()
    token = os.getenv('X_BEARER_TOKEN', '').strip()
    if not token:
        sys.exit('X_BEARER_TOKEN belum diisi pada .env. Baca README.md.')

    entries, seen = [], set()
    next_token = None
    session = requests.Session()
    while len(entries) < args.target:
        params = {
            'query': args.query, 'max_results': 100,
            'tweet.fields': 'created_at,author_id',
            'expansions': 'author_id', 'user.fields': 'username',
        }
        if next_token:
            params['next_token'] = next_token
        try:
            resp = session.get(ENDPOINT, params=params,
                               headers={'Authorization': f'Bearer {token}'}, timeout=40)
            resp.raise_for_status()
        except requests.RequestException as error:
            msg = getattr(error.response, 'text', '')[:500] if getattr(error, 'response', None) is not None else ''
            sys.exit(f'Permintaan X API gagal. Cek paket akses, kuota, atau token: {error}\n{msg}')
        payload = resp.json()
        users = {str(u['id']): u.get('username', '') for u in payload.get('includes', {}).get('users', [])}
        for tweet in payload.get('data', []):
            if tweet['id'] in seen:
                continue
            seen.add(tweet['id'])
            entries.append({
                'created_at': tweet.get('created_at', ''),
                'username': users.get(str(tweet.get('author_id', '')), ''),
                'full_text': tweet.get('text', ''),
                'source': 'X', 'source_url': f"https://x.com/i/web/status/{tweet['id']}"
            })
            if len(entries) >= args.target:
                break
        print(f'Diperoleh {len(entries)} posting...')
        next_token = payload.get('meta', {}).get('next_token')
        if not next_token:
            break
        time.sleep(0.5)

    df = pd.DataFrame(entries, columns=['created_at', 'username', 'full_text', 'source', 'source_url'])
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Selesai: {len(df)} posting asli disimpan di {args.output}')
    if len(df) < 300:
        print('PERHATIAN: Belum mencapai 300 data. Coba perluas query atau periode jika akses memungkinkan.')


if __name__ == '__main__':
    main()
