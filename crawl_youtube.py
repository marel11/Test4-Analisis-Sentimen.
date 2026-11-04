"""Alternatif API resmi YouTube: kumpulkan komentar video mitigasi bencana."""
import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

API = 'https://www.googleapis.com/youtube/v3'
SEARCHES = ['mitigasi bencana Indonesia BNPB', 'kesiapsiagaan bencana gempa Indonesia', 'edukasi evakuasi bencana Indonesia']


def get_json(session, endpoint, params):
    try:
        response = session.get(f'{API}/{endpoint}', params=params, timeout=35)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        detail = (error.response.text[:600] if getattr(error, 'response', None) is not None else '')
        sys.exit(f'YouTube API gagal: {error}\n{detail}')


def main():
    parser = argparse.ArgumentParser(description='Unduh komentar YouTube terkait mitigasi bencana')
    parser.add_argument('--target', type=int, default=350)
    parser.add_argument('--output', default='tweet_data.csv', help='Nama file kompatibilitas tugas; kontennya komentar YouTube')
    args = parser.parse_args()
    load_dotenv()
    key = os.getenv('YOUTUBE_API_KEY', '').strip()
    if not key:
        sys.exit('YOUTUBE_API_KEY belum diisi pada .env. Lihat README.md.')
    session = requests.Session()
    ids = []
    for term in SEARCHES:
        data = get_json(session, 'search', {
            'key': key, 'part': 'snippet', 'q': term, 'maxResults': 25,
            'type': 'video', 'relevanceLanguage': 'id', 'order': 'relevance'
        })
        for video in data.get('items', []):
            vid = video.get('id', {}).get('videoId')
            if vid and vid not in ids:
                ids.append(vid)
    print(f'Video relevan ditemukan: {len(ids)}')
    rows, seen = [], set()
    for vid in ids:
        token = None
        while len(rows) < args.target:
            params = {
                'key': key, 'part': 'snippet', 'videoId': vid,
                'maxResults': 100, 'textFormat': 'plainText', 'order': 'relevance'
            }
            if token:
                params['pageToken'] = token
            try:
                result = get_json(session, 'commentThreads', params)
            except SystemExit as error:
                # Komentar video tertentu dapat ditutup; lanjutkan video lainnya.
                print(f'Lewati video {vid}: {error}')
                break
            for item in result.get('items', []):
                top = item.get('snippet', {}).get('topLevelComment', {})
                cid = top.get('id')
                comment = top.get('snippet', {})
                txt = comment.get('textDisplay', '').strip()
                if not cid or cid in seen or len(txt) < 15:
                    continue
                seen.add(cid)
                rows.append({
                    'created_at': comment.get('publishedAt', ''),
                    'username': comment.get('authorDisplayName', ''),
                    'full_text': txt, 'source': 'YouTube',
                    'source_url': f'https://www.youtube.com/watch?v={vid}&lc={cid}',
                })
                if len(rows) >= args.target:
                    break
            token = result.get('nextPageToken')
            if not token:
                break
            time.sleep(0.15)
        print(f'Komentar terkumpul: {len(rows)}')
        if len(rows) >= args.target:
            break
    df = pd.DataFrame(rows, columns=['created_at', 'username', 'full_text', 'source', 'source_url'])
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Selesai: {len(df)} komentar asli disimpan di {args.output}')
    if len(df) < 300:
        print('PERHATIAN: Data kurang dari 300. Perbanyak video/kueri dan periksa relevansi manual.')
    else:
        print('PERHATIAN: Video relevan tidak menjamin seluruh komentar relevan. Periksa komentar sebelum analisis.')


if __name__ == '__main__':
    main()
