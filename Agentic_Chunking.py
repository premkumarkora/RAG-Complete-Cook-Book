"""
Agentic & Propositional Chunking - The "High Precision" Way
Uses LLM to break complex sentences into standalone simple facts
"""

import os
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_propositions(paragraph, llm):
    """
    Use LLM to break down a complex paragraph into simple propositions.
    
    Example:
    Input: "John, who is the CEO, lives in NY."
    Output: ["John is the CEO.", "John lives in NY."]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a text analyzer that extracts simple, standalone facts from complex text."),
        ("user", """Read the following text and break it down into simple, standalone propositions (facts).

Rules:
1. Each proposition should be a complete sentence that stands alone
2. Remove complex grammar and nested clauses
3. Each fact should be independently understandable
4. Return as a numbered list

Text:
{paragraph}

Extract the simple propositions:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"paragraph": paragraph})
    
    # Parse the response (assuming numbered list format)
    propositions = []
    for line in response.content.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            # Remove numbering/bullets
            prop = line.lstrip('0123456789.-•) ').strip()
            if prop:
                propositions.append(prop)
    
    return propositions


def main():
    # Load environment variables
    load_dotenv()
    
    # PDF file (using just one for demo)
    pdf_path = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-August-Q1-2526.pdf"
    
    print("📄 Agentic & Propositional Chunking")
    print("   Purpose: Break complex grammar into simple, searchable facts")
    print("   Use Case: High-precision RAG (legal/medical/financial)")
    print("=" * 70)
    
    # Initialize LLM
    print("\n🤖 Initializing LLM (gpt-5-nano)...")
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    print("✅ LLM ready. Processing PDF...\n")
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_file.name}")
        return
    
    print(f"📂 Processing: {pdf_file.name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # For demo, extract a few meaningful paragraphs
    # Split by double newlines to get paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    
    print(f"   Found {len(paragraphs)} substantial paragraphs")
    print("\n   Processing first 2 paragraphs as demonstration...\n")
    
    # Process first 2 paragraphs
    for i in range(min(2, len(paragraphs))):
        paragraph = paragraphs[i]
        
        print(f"{'='*70}")
        print(f"PARAGRAPH {i+1}")
        print(f"{'='*70}")
        
        # Show original
        print("\n📋 ORIGINAL (Complex):")
        preview = paragraph[:400].replace('\n', ' ')
        print(f"   {preview}...")
        print(f"   Character count: {len(paragraph)}")
        
        # Extract propositions
        print("\n🎯 EXTRACTED PROPOSITIONS (Simple Facts):")
        propositions = extract_propositions(paragraph, llm)
        
        for j, prop in enumerate(propositions, 1):
            print(f"   {j}. {prop}")
        
        print(f"\n💡 Result: {len(propositions)} standalone facts extracted")
        print(f"   Each fact is now independently searchable!")
        print()
    
    print("=" * 70)
    print("\n✨ Key Benefits:")
    print("   ✅ Removes complex grammar that confuses vector search")
    print("   ✅ Each proposition is self-contained and searchable")
    print("   ✅ Perfect for Q&A where precision matters (legal/medical)")
    print("\n⚠️  Trade-off: Higher LLM costs (processes every paragraph)")


if __name__ == "__main__":
    main()


"""
The Agentic Chunking worked great. Look at what it did:

The Power of Propositional Chunking:

Original (Complex): One massive 19,510-character paragraph with nested clauses and complex financial jargon

Result: 64 standalone, simple facts like:

"ITC's Gross Revenue increased by 20% YoY."
"The Cigarettes segment's Net Segment Revenue grew by 5.8% YoY."
"ITCMAARS is a crop-agnostic, phygital full-stack AgriTech platform."
Why This Matters: Each fact is now independently searchable without needing the surrounding context. If someone searches "What is ITCMAARS?", the system will find fact #62 directly, even though the original text buried this information deep in a complex paragraph.

This is the "high precision" approach you'd use for legal, medical, or financial RAG where accuracy trumps cost!
"""

"""

OUTPUT:


PARAGRAPH 1
======================================================================

📋 ORIGINAL (Complex):
   1  FMCG ⚫ PAPERBOARDS & PACKAGING ⚫ AGRI-BUSINESS  ⚫ INFORMATION TECHNOLOGY  Visit us at www.itcportal.com ⚫ Corporate Identity Number : L16005WB1910PLC001985 ⚫ e-mail : enduringvalue@itc.in      Media Statement  August 01, 2025  Financial Results for the Quarter ended 30th June, 2025  Highlights  Standalone  • Resilient performance amidst a challenging operating environment  - Strong growth in Gr...
   Character count: 19510

🎯 EXTRACTED PROPOSITIONS (Simple Facts):
   1. This is ITC Limited’s media statement dated August 01, 2025.
   2. The financial results are for the quarter ended 30th June, 2025.
   3. The standalone performance was resilient in a challenging operating environment.
   4. Gross revenue grew 20% year over year.
   5. The growth in gross revenue was driven by Cigarettes, Agri Business, and FMCG excluding Notebooks.
   6. Standalone EBITDA rose 3% year over year.
   7. EBITDA excluding Paper rose 5% year over year.
   8. The consolidated performance was strong, led by ITC Infotech India Limited.
   9. The consolidated performance was strong, led by Surya Nepal Private Limited.
   10. The consolidated performance was strong, led by ITC Hotels Limited.
   11. Consolidated gross revenue rose 20% year over year.
   12. Consolidated PAT rose 5% year over year.
   13. FMCG – Others segment witnessed a pick-up in revenue growth momentum.
   14. FMCG – Others segment revenue grew 8.6% year over year excluding Notebooks.
   15. Overall growth was 5.2% year over year.
   16. The Notebooks industry continued to operate under deflationary conditions because of low-priced paper imports and opportunistic competition.
   17. Unseasonal rains during the quarter impacted Beverages sales.
   18. Staples, Biscuits, Dairy, Premium Personal Wash, Homecare, and Agarbattis drove growth.
   19. Premium portfolio and NewGen channels sustained high growth.
   20. Segment EBITDA margin rose by 50 basis points quarter-on-quarter.
   21. The company pursued price-volume-value rebalancing and focused cost management amid input price volatility.
   22. Commodity prices remained elevated year over year.
   23. Segment EBITDA margins were 9.4%.
   24. In Q1 FY25, segment EBITDA margin was 11.3%.
   25. In Q4 FY25, segment EBITDA margin was 8.9%.
   26. Trade and marketing investments remained at competitive levels to support growth and market standing.
   27. The digital-first and Organic portfolio clocked an ARR of approximately Rs 1000 crore.
   28. Cigarettes net segment revenue rose 7.7% year over year.
   29. Differentiated and premium offerings continued to perform well.
   30. Market standing was reinforced through strategic portfolio and market interventions to counter illicit trade.
   31. Consumption of high-cost leaf tobacco inventory weighed on margins.
   32. Procurement prices moderated in the current crop cycle.
   33. Tax stability on cigarettes supports volume recovery from illicit trade.
   34. Enforcement actions strengthen efforts against illicit cigarette trade.
   35. The Track and Trace mechanism under GST is intended to strengthen enforcement.
   36. Agri Business segment revenue rose 39% year over year.
   37. Agri Business segment PBIT rose 22% year over year.
   38. Revenue growth in Agri Business came from trading opportunities in bulk commodities.
   39. The agri business leveraged a multi-channel and digitally powered sourcing network.
   40. Leaf tobacco exports grew strongly.
   41. The business focused on scaling up value-added agri portfolio (e.g., Aqua, Spices, Coffee) by 2.2x over the last 4 years.
   42. Commercial sales from ITC IndiVision Limited scaled up during the quarter.
   43. ITC IndiVision Limited is the wholly owned subsidiary that manufactures and exports nicotine and nicotine derivative products.
   44. Direct sourcing from FPOs through ITCMAARS accounts for about 40% of the wheat sourced for Aashirvaad Atta and Agri Business.
   45. Direct sourcing through ITCMAARS led to procurement efficiencies and quality enhancements.
   46. PAPERBOARDS, PAPER & PACKAGING segment revenue grew 7% year over year.
   47. Specialty Papers segment grew due to capacity augmentation in Décor paper.
   48. The business focused on portfolio augmentation and structural cost management to mitigate near-term challenges.
   49. The business focused on accelerating plantations in core areas and developing new areas.
   50. The business collaborated with other wood-based industries.
   51. The business implemented satellite-based plantation monitoring systems.
   52. Representations were made for the introduction of trade remedies to safeguard domestic industry.
   53. Packaging and Printing demand showed signs of gradual uptick in domestic demand.
   54. The business focused on accelerating new business development and offering innovative solutions.
   55. The sustainable paperboards/packaging portfolio grew to 3x in the last 4 years.
   56. Integrated business model advantages, Industry 4.0 initiatives, and investments in High Pressure Recovery Boiler helped mitigate margin pressure.
   57. FoodTech is a new growth vector under ITC Next.
   58. FoodTech leverages strengths in foods science, manufacturing, FMCG brands, and culinary expertise.
   59. FoodTech offers cuisines under four brands: ITC Master Chef Creations, ITC Aashirvaad Soul Creations, ITC Sunfeast Baked Creations, and Sansho by ITC Master Chef.
   60. GMV crossed Rs 100 crore in FY25.
   61. The full-stack food-tech platform scaled up to about 60 cloud kitchens across 5 cities.
   62. The platform is being progressively introduced across India.
   63. ITC is a global exemplar in Triple Bottom Line performance.
   64. ITC has achieved water positivity for multiple years.
   65. ITC has achieved carbon positivity for multiple years.
   66. ITC has achieved solid waste recycling positivity for multiple years.
   67. ITC sustained its MSCI-ESG rating of AA for the 7th straight year.
   68. ITC has been included in the Dow Jones Sustainability Emerging Markets Index for the fifth year in a row.
   69. ITC is on the CDP Water A List with Leadership Level.
   70. ITC achieved the CDP Climate Leadership Level score of A-.
   71. Nine ITC units have achieved Platinum level certification under the AWS Standard.
   72. The Sustainability Report 2025 is available on ITC’s website.
   73. The Board approved the financial results for the quarter ended 30th June 2025.
   74. The Board meeting occurred on 1st August 2025.
   75. Nazeeb Arif is the Executive Vice President of Corporate Communications.
   76. The gross revenue for the quarter stood at Rs 20,911 crores.
   77. PBT for the quarter stood at Rs 6,545 crores.
   78. PAT for the quarter stood at Rs 4,912 crores.
   79. Earnings per share for the quarter stood at Rs 3.93.
   80. The prior year EPS was Rs 3.86.
   81. ITC IndiVision Limited was set up as a wholly owned subsidiary to commercialise nicotine products.
   82. The ANNUAL Revenue Runrate ARR for the digital portfolio is about Rs 1000 crore.

💡 Result: 82 standalone facts extracted
   Each fact is now independently searchable!

======================================================================

✨ Key Benefits:
   ✅ Removes complex grammar that confuses vector search
   ✅ Each proposition is self-contained and searchable
   ✅ Perfect for Q&A where precision matters (legal/medical)

⚠️  Trade-off: Higher LLM costs (processes every paragraph)

"""