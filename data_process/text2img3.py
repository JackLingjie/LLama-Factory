from markdown import markdown  
import tempfile
import subprocess
from html2image import Html2Image  
import os  
import logging  
  
# 配置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def markdown_to_html(input_text):  
    # 使用临时文件存储 Markdown 和 HTML  
    with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as md_file:  
        md_file.write(input_text.encode('utf-8'))  
        md_file_path = md_file.name  
  
    html_file_path = md_file_path.replace('.md', '.html')  
  
    # 构建 pandoc 命令  
    pandoc_command = [  
        'pandoc',  
        '-s',  
        md_file_path,  
        '-o', html_file_path,  
        '--self-contained'  
    ]  
  
    # 运行 pandoc 命令  
    try:  
        subprocess.run(pandoc_command, check=True)  
        print(f'转换成功: {html_file_path}')  
    except subprocess.CalledProcessError as e:  
        print(f'转换失败: {e}')  
        return None  
    print(html_file_path)
    return html_file_path
  
def text_to_image(text, output_image, size=(800, 600), save_dir='text_images'):  
    """Convert text to image and log the process"""  
    logging.info("Starting the conversion process.")  
      
    # Convert text to Markdown (assume the text is in Markdown format)  
    html_file_path = markdown_to_html(text)  
      
    # Wrap HTML header and style  
    # html_content = f"""  
    # <!DOCTYPE html>  
    # <html lang="en">  
    # <head>  
    #     <meta charset="UTF-8">  
    #     <title>Markdown to HTML Example</title>  
    #     <style>  
    #         body {{  
    #             font-family: 'Arial', sans-serif;  
    #             margin: 0;  
    #             padding: 20px;  
    #             background-color: white;  
    #             color: black;  
    #         }}  
    #     </style>  
    # </head>  
    # <body>  
    #     {html_content}  
    # </body>  
    # </html>  
    # """  
    output_path = os.path.dirname(os.path.abspath(__file__)) 
    output_path = os.path.join(output_path, save_dir)
    # Convert HTML to image using html2image  
    hti = Html2Image(size=(800, 600), temp_path="/home/lidong1/jianglingjie/temp", output_path=output_path)  

    # 设置浏览器路径（如果需要）  
    # hti.browser_executable = '/path/to/your/chrome-or-edge'  
     # Get the directory of the current script  
    # script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # output_image = os.path.join(script_dir, output_image)    
    # output_image = 'data_process/red_page.png'
    # logging.info(f"Image saved to {output_image}.")    
    # hti.screenshot(html_str=html_content, save_as=output_image)  
    hti.screenshot(html_file=html_file_path, save_as=output_image)  
      
    logging.info(f"Image saved to {os.path.join(output_path, output_image)}.")  
      
    logging.info("Conversion process completed successfully.")  
  
markdown_text = \
"""Developing a daily habit of drawing can be challenging but with consistent practice and a few tips, it can become an enjoyable and rewarding part of your daily routine. Here are some strategies to help you develop the habit of drawing daily:

1. Set a specific time: Allocate a specific time of the day to draw. It could be in the morning, afternoon, or evening. Make drawing a part of your daily routine.
2. Set a specific duration: Determine the amount of time you want to spend on drawing each day. It can be as little as 10 minutes or as long as an hour. Be consistent with the duration to help build the habit.
3. Start small and simple: Don't try to create a masterpiece every day, start with simple and easy-to-do sketches. Focus on improving your skills gradually.
4. Use a variety of tools and mediums: Experiment with different tools like pencils, pens, markers, and different mediums like paper, canvas, or digital apps to keep your drawing practice interesting and engaging.
5. Take breaks and rest: Taking breaks and resting after some time of drawing can help you avoid burnout and stay motivated.
6. Challenge yourself: Set challenges like drawing objects from memory or a specific subject to improve your skills and keep your drawing practice interesting.
7. Track your progress: Keep a record of your daily drawing practice and track your progress. This can be a source of motivation and help you see how far you've come.

Remember, developing a habit takes time and patience. Stay consistent with your drawing practice, be flexible and open to trying new things, and with time, you'll develop a habit of daily drawing that brings you joy and satisfaction.  
"""  
  
if __name__ == "__main__":  
    # 示例Markdown文本  
      
    # 调用函数并输出日志  
    output_image = 'output.png'  
    text_to_image(markdown_text, output_image)  
