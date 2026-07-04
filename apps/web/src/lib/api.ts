const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type HealthResponse = {
  service: string;
  status: "ok";
};

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/v1/health`);

  if (!response.ok) {
    throw new Error(`API health check failed: ${response.status}`);
  }

  return response.json() as Promise<HealthResponse>;
}
