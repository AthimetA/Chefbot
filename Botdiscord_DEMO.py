import discord
import numpy as np
import re
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.models import load_model

def prepare_input_for_discord(input_string):

    # Regular expression to match phone numbers
    pattern = r"\d{3}[-\s]?\d{3}[-\s]?\d{4}"
    
    # Replace phone numbers with PhoneNumber
    input_string = re.sub(pattern, "PhoneNumber", input_string)

    # Remove unnecessary characters
    input_string = input_string.strip()
    input_string = input_string.replace(" ", "")
    input_string = input_string.replace("\n", "")
    input_string = input_string.replace("!", "")
    input_string = input_string.replace("ค่ะ", "")
    input_string = input_string.replace("จะ", "")
    input_string = input_string.replace("ครับ", "")
    input_string = input_string.replace("ผม", "")
    input_string = input_string.replace("คะ", "")
    input_string = input_string.replace("อ่ะ", "")
    input_string = input_string.replace("พี่", "")
    input_string = input_string.replace("เรา", "")
    input_string = input_string.replace("เค้า", "")

    action_keywords = ['เปลี่ยน', 'โปรโมชั่น', 'เป็น', 'ที่อยู่', 'แจ้ง','อะไร', 'คือ', 'หนู', 'เมื่อกี้', 'สอบถาม', 'เครื่อง', 'มี', 
                    'ช็อป', 'ทราบ', 'ทรู', 'ของ', 'เปิด', 'หน่อย', 'ใช้', 'ชำระ', 'จ่าย','สมัคร', 'อินเตอร์เน็ต', 'โปร', 'เน็ต', 'บาท', 'แพคเกจ', 
                    'บี', 'ซื้อ', 'ต้อง', 'รายวัน', 'วัน','ยกเลิก', 'ข้อความ', 'ที่', 'อยาก', 'ขอ', 'ต้องการ', 'SMS', 'บริการ', 'มา', 'สาย', 'ให้', 
                    'รอ', 'นี้', 'โทรศัพท์', 'ระงับ','โดน','ซิม','ออก']
    
    result = np.zeros(len(action_keywords))
    for index, keyword in enumerate(action_keywords):
        if keyword in input_string:
            result[index] = 1

    result = result.reshape(1, 50)

    return result

def predict_action(input_string):

    action_label = ['enquire', 'report', 'cancel', 'buy', 'activate', 'request','garbage', 'change']
    num_to_action_label_map = dict(zip(range(len(action_label)), action_label))

    # Load the model from file
    model_action = load_model('model_action.h5')

    # Prepare input for the model
    input_after = prepare_input_for_discord(input_string)

    # Use the model for prediction
    prediction = model_action.predict(input_after)

    # Get predicted class label
    num_to_action_label_map = dict(zip(range(len(action_label)), action_label))
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = num_to_action_label_map[predicted_class_index]

    return predicted_class_label

bot = discord.Client()
TOKEN = "MTEwMzI1OTEyMTE2OTQ3MzU1Ng.Gcj3Je.dLKOCtiC8Ebe4NjZTIyXSs8xwJNSFIJ8bKbf-U"

@bot.event
async def on_ready():
    print("Bot Started")

@bot.event
async def on_message(message):
    if message.author == bot.user:      
        return 
    if f'{bot.user.mention}' in message.content:
        input_string = message.content
        input_string = input_string.replace(f'{bot.user.mention}', '')
        input_string = input_string.replace(' ', '')
        predicted_class_label = predict_action(message.content)
        await message.channel.send(f"{predicted_class_label}")

if __name__ == '__main__':
    bot.run(TOKEN)