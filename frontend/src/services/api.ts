/**
 * Backend API client.
 *
 * The FastAPI backend will be connected here.
 */

const API_BASE_URL = "http://localhost:8000";

export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("Backend unavailable");
  }

  return response.json();
}
