import sys
import requests
import json

URL = 'https://www.sms4india.com/api/v1/sendCampaign'



# get request
def sendPostRequest(reqUrl, textMessage):
    req_params = {
    'apikey':'PUWQKV874IGJNYKNVPNYT9ACTJQ88TM5',
    'secret':'L7KWCFL3QFCVN4VJ',
    'usetype':'stage',
    'phone': '9164883464',
    'message': textMessage,
    'senderid':'senderId'
    }
    return requests.post(reqUrl, req_params)

# get response
def send_sms(amount, beneficiary):
    body = "Your a/c no. XXXXXXXX6934 is debited for Rs, " + str(amount) + " on 02-02-20 and " + beneficiary +  " credited (IMPS Ref no 003310698407)."
    response = sendPostRequest(URL, body)
    print (response.text)