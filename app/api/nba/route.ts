import { NextRequest, NextResponse } from 'next/server';
const ALLOWED_ENDPOINTS = new Set(['playertotals', 'playeradvancedstats']);
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const endpoint  = searchParams.get('endpoint')  ?? 'playertotals';
  const season    = searchParams.get('season')     ?? '2016';
  const isPlayoff = searchParams.get('isPlayoff')  ?? 'false';
  const page      = searchParams.get('page')       ?? '1';
  const pageSize  = searchParams.get('pageSize')   ?? '100';
  if (!ALLOWED_ENDPOINTS.has(endpoint)) {
    return NextResponse.json(
      { error: `Endpoint "${endpoint}" is not permitted.` },
      { status: 400 }
    );
  }
  const base = process.env.NBA_API_BASE ?? 'https://api.server.nbaapi.com/api';
  const upstreamUrl =
    `${base}/${endpoint}` +
    `?season=${encodeURIComponent(season)}` +
    `&isPlayoff=${encodeURIComponent(isPlayoff)}` +
    `&pageSize=${encodeURIComponent(pageSize)}` +
    `&page=${encodeURIComponent(page)}`;
  try {
    const upstreamRes = await fetch(upstreamUrl);
    if (!upstreamRes.ok) {
      return NextResponse.json(
        {
          error: `NBA API returned ${upstreamRes.status}: ${upstreamRes.statusText}`,
        },
        { status: upstreamRes.status }
      );
    }
    const data = await upstreamRes.json();
    return NextResponse.json(data, {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown fetch error';
    return NextResponse.json(
      { error: `Proxy failed to reach NBA API: ${message}` },
      { status: 502 }
    );
  }
}
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin':  '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}