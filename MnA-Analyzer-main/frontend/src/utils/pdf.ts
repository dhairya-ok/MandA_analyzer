const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export async function downloadDealBrief(pairId: string): Promise<void> {
    try {
        const response = await fetch(`${apiBase}/api/deal-brief/${pairId}`);
        if (!response.ok) throw new Error(await response.text());
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `deal_brief.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Failed to download brief:', error);
        throw error;
    }
}