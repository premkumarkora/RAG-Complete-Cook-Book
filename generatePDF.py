from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap

def create_tricky_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Confidential Technology History Notes")

    c.setFont("Helvetica", 12)
    y_position = height - 120

    # --- TRICKY PART 1: The Code Name ---
    # This section mentions the secret name but not the final product or person.
    text_block_1 = """
    [Internal Memo - 2005]
    The secret initiative known internally as 'Project Purple' has received approval to move into the prototype phase. 
    It is critical that the true nature of this revolutionary touchscreen interface remains completely confidential outside the design lab. 
    Failure to maintain secrecy will result in immediate termination.
    """
    
    lines = textwrap.wrap(text_block_1.strip(), width=80)
    for line in lines:
        c.drawString(72, y_position, line)
        y_position -= 15

    y_position -= 30 # Add spacing

    # --- TRICKY PART 2: The Person and Product ---
    # This section mentions the person and final product, but not the code name.
    text_block_2 = """
    [Industry News Clipping - 2007]
    Apple is betting big on its upcoming iPhone device. 
    Key sources indicate that executive Scott Forstall was chosen to lead the critical software development team, pushing for the multitouch-based operating system over competing hardware-keyboard designs.
    """

    lines = textwrap.wrap(text_block_2.strip(), width=80)
    for line in lines:
        c.drawString(72, y_position, line)
        y_position -= 15

    c.save()
    print(f"Successfully created: {filename}")

if __name__ == "__main__":
    create_tricky_pdf("apple_secret_history.pdf")