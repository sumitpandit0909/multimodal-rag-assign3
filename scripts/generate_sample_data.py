import os
from pathlib import Path
import fitz  # PyMuPDF
import openpyxl

DATA_DIR = Path(__file__).resolve().parents[1] / "data_drop"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_pdf(filename: str, slides_content: list):
    pdf_path = DATA_DIR / filename
    doc = fitz.open()
    
    for slide in slides_content:
        # Standard 16:9 presentation slide dimensions (960 x 540 points)
        page = doc.new_page(width=960, height=540)
        
        # Draw dark aesthetic background
        rect = fitz.Rect(0, 0, 960, 540)
        page.draw_rect(rect, color=(0.08, 0.09, 0.13), fill=(0.08, 0.09, 0.13))
        
        # Header banner line
        page.draw_line(fitz.Point(60, 110), fitz.Point(900, 110), color=(0.4, 0.45, 0.95), width=2)
        
        # Slide Title
        page.insert_text(fitz.Point(60, 85), slide["title"], fontsize=28, color=(1, 1, 1))
        
        # Slide Content Bullet Points
        y = 170
        for bullet in slide["bullets"]:
            page.insert_text(fitz.Point(70, y), f"•  {bullet}", fontsize=18, color=(0.85, 0.88, 0.92))
            y += 45
            
        # Footer
        page.insert_text(fitz.Point(60, 500), f"Confidential Enterprise Knowledge Base | {filename}", fontsize=10, color=(0.4, 0.45, 0.55))
        
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created presentation PDF: {pdf_path.name}")

def create_excel(filename: str):
    wb = openpyxl.Workbook()
    
    # Sheet 1: Casino Performance
    ws1 = wb.active
    ws1.title = "Casino_Gaming_Revenue"
    ws1.append(["Quarter", "Gaming Floor", "VIP Club Revenue ($M)", "Slots Revenue ($M)", "Gross Gaming Margin"])
    ws1.append(["2025-Q1", "Las Vegas Strip", 145.2, 88.4, "41.2%"])
    ws1.append(["2025-Q2", "Las Vegas Strip", 156.8, 94.1, "43.5%"])
    ws1.append(["2025-Q3", "Macau Integrated Resort", 210.5, 112.3, "46.8%"])
    ws1.append(["2025-Q4", "Macau Integrated Resort", 225.0, 118.7, "47.2%"])
    
    # Sheet 2: VIX Micro Metrics
    ws2 = wb.create_sheet(title="VIX_Micro_Hedging")
    ws2.append(["Trading Date", "Contract", "Implied Volatility", "Average Daily Volume", "Hedge Ratio"])
    ws2.append(["2025-10-01", "VIX Micro Nov25", "16.4%", 45200, 0.35])
    ws2.append(["2025-10-15", "VIX Micro Nov25", "18.2%", 51400, 0.42])
    ws2.append(["2025-11-01", "VIX Micro Dec25", "21.5%", 68900, 0.55])
    
    excel_path = DATA_DIR / filename
    wb.save(str(excel_path))
    print(f"Created Excel workbook: {excel_path.name}")

def main():
    # 1. Deck for Casinos
    create_pdf("Casino_Operations_Presentation.pdf", [
        {
            "title": "Integrated Casino & Gaming Resort Strategy",
            "bullets": [
                "Comprehensive operations deck for modern casino resorts and high-roller VIP lounges.",
                "Overview of luxury table games (Baccarat, Blackjack, Roulette) and electronic gaming machines (Slots).",
                "Hospitality synergies: Michelin-starred dining, luxury suites, and entertainment residencies.",
                "Cash flow optimization: High Net Worth (HNW) patron credit lines and anti-money laundering compliance."
            ]
        },
        {
            "title": "Digital Gaming & Casino Loyalty Systems",
            "bullets": [
                "Omnichannel rewards combining on-property casino perks with digital igaming points.",
                "Player Tracking Systems (PTS) for automated comps and real-time floor telemetry.",
                "Regulatory licensing framework for casino gaming commissions."
            ]
        }
    ])
    
    # 2. Content deck for VIX Micro
    create_pdf("VIX_Micro_Financial_Overview.pdf", [
        {
            "title": "VIX Micro Futures & Volatility Management",
            "bullets": [
                "Executive content deck introducing VIX Micro (1/10th index size) volatility contracts.",
                "Precision risk hedging for institutional portfolios during equity market drawdowns.",
                "Capital efficiency: Lower margin requirements compared to standard Cboe VIX futures.",
                "Term structure dynamics: Contango vs backwardation roll-yield analysis."
            ]
        }
    ])
    
    # 3. Slide for Bilingualism & Biculturalism
    create_pdf("Bilingualism_and_Biculturalism_Study.pdf", [
        {
            "title": "Foundations of Bilingualism & Biculturalism",
            "bullets": [
                "Cognitive advantages of bilingualism: Executive function, memory retention, and divergent thinking.",
                "Bicultural identity integration: Navigating dual linguistic norms and sociolinguistic codes.",
                "Pedagogical frameworks for immersion classrooms and heritage language development.",
                "Cross-cultural communication strategies for global enterprises."
            ]
        }
    ])
    
    # 4. Tabular Excel
    create_excel("Corporate_Data_Q4.xlsx")
    print("All sample datasets generated in data_drop/!")

if __name__ == "__main__":
    main()
