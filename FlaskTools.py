# Get openCV image from Flask Request:
import cv2
from flask import request

def request_to_opencv(request):
    open_cv_image = cv2.imdecode(np.frombuffer(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    return open_cv_image
    
#########################################################################################################################

# Return as API Rest:

from flask import jsonify
@app.route('/API', methods=['GET', 'POST'])
def getRequest():
  #process
  return jsonify({'Some text': some_variable})
