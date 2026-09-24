export async function POST(request: Request) {
  const body = await request.text();
  const upstream = process.env.RAG_API_URL ?? "http://127.0.0.1:8000/ask";
  try {
    const response = await fetch(upstream, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const payload = await response.text();
    return new Response(payload, {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json(
      {
        error: "The policy service is not running.",
        answer: "",
        citations: [],
      },
      { status: 502 },
    );
  }
}
