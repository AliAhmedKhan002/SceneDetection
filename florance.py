from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from matplotlib import patches, pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import torch
from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import base64

app = FastAPI()

# model_id = 'Florence-2-large'
model_id = 'Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    # plt.show()  
    plt.savefig('/home/ali/AliAhmed/Florance/Test_Images/test.png')
    

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    torch.cuda.empty_cache()
    task_prompt = '<OD>'
    image = Image.open(io.BytesIO(await file.read()))
    results = run_example(image, task_prompt)
    plot_bbox(image, results['<OD>'])
    # results = run_example(image, task_prompt)
    # text_input = results[task_prompt]
    # task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    # results1 = run_example(image, task_prompt, text_input)
    # results1['<MORE_DETAILED_CAPTION>'] = text_input
    # plot_bbox(image, results1['<CAPTION_TO_PHRASE_GROUNDING>'])

    return JSONResponse(content={
        "results": results, 
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("florance:app", host="192.168.18.164", port=8007, reload=True)
