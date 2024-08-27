from selenium import webdriver  
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.chrome.options import Options  
from PIL import Image  
import os  
import markdown  
import logging  
  
# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def markdown_to_html(markdown_text):  
    """Convert Markdown text to HTML"""  
    html = markdown.markdown(markdown_text)  
    return html  
  
def save_html_to_file(html_content, file_path):  
    """Save HTML content to a file"""  
    with open(file_path, 'w', encoding='utf-8') as file:  
        file.write(html_content)  
  
def html_to_image(html_file, output_image):  
    # Set Chrome options  
    chrome_options = Options()  
    chrome_options.add_argument("--headless")  # Headless mode  
    chrome_options.add_argument("--disable-gpu")  
    chrome_options.add_argument("--window-size=1920x1080")  
    chrome_options.add_argument("--no-sandbox")  
      
    # Set ChromeDriver service  
    service = Service('/usr/local/bin/chromedriver')  # Path to ChromeDriver  
      
    # Start Chrome browser  
    driver = webdriver.Chrome(service=service, options=chrome_options)  
      
    # Open HTML file  
    file_url = f"file://{os.path.abspath(html_file)}"  
    driver.get(file_url)  
      
    # Take screenshot  
    screenshot = driver.get_screenshot_as_png()  
      
    # Save image  
    with open(output_image, 'wb') as file:  
        file.write(screenshot)  
      
    # Close browser  
    driver.quit()  
      
    # Use PIL to adjust image size  
    with Image.open(output_image) as img:  
        img = img.crop(img.getbbox())  # Crop blank areas  
        img.save(output_image)  
  
def text_to_image_with_log(text, output_image):  
    """  
    Convert text to image and log the process  
    """  
    logging.info("Starting the conversion process.")  
      
    # Convert text to Markdown (assume the text is in Markdown format)  
    html_content = markdown_to_html(text)  
      
    # Wrap HTML header and style  
    html_content = f"""  
    <!DOCTYPE html>  
    <html lang="en">  
    <head>  
        <meta charset="UTF-8">  
        <title>Markdown to HTML Example</title>  
        <style>  
            body {{  
                font-family: 'Arial', sans-serif;  
            }}  
        </style>  
    </head>  
    <body>  
        {html_content}  
    </body>  
    </html>  
    """  
      
    # Save HTML to file  
    html_file = 'example.html'  
    save_html_to_file(html_content, html_file)  
    logging.info(f"HTML content saved to {html_file}.")  
      
    # Convert HTML to image  
    html_to_image(html_file, output_image)  
    logging.info(f"Image saved to {output_image}.")  
      
    # Delete HTML file  
    os.remove(html_file)  
    logging.info(f"Temporary HTML file {html_file} deleted.")  
      
    logging.info("Conversion process completed successfully.")  
def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = "revised_data/output.jsonl"  
    output_file = os.path.join(cur_dir, output_file)
    with open(output_file, 'r') as f:
        data = f.read()
        print(data)
    # markdown_text = """  
    # # Hello, World!  

    # This is an example document written using **Markdown**.  

    # - Item 1  
    # - Item 2  
    # - Item 3  

    # [Click here](http://example.com) to visit the example website.  
    # """  

    # # 调用函数并输出日志  
    # output_image = 'output.png'  
    # text_to_image_with_log(markdown_text, output_image)  
main()
# if __name__ == "__main__":
    # 示例Markdown文本  
# markdown_text = """  
# # Hello, World!  

# This is an example document written using **Markdown**.  

# - Item 1  
# - Item 2  
# - Item 3  

# [Click here](http://example.com) to visit the example website.  
# """  

# # 调用函数并输出日志  
# output_image = 'output.png'  
# text_to_image_with_log(markdown_text, output_image)  
