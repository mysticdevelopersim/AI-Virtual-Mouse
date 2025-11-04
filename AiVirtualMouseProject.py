import cv2
import numpy as np
import HandTrackingModule as htm  # Ensure this module is correct and available
import time
import autopy

# Webcam settings
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 10  # Smoothing factor (increase for smoother movement)

# Initialize previous time for FPS calculation
pTime = 0
plocX, plocY = 0, 0  
clocX, clocY = 0, 0
# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize the hand detector
detector = htm.handDetector(maxHands=1)

# Get screen width and height
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find the Hand Landmarks
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image from webcam.")
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the Tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # Draw the frame boundary
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))  # Map to screen width
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))  # Map to screen height

            # 6. Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)  # Adjust for screen coordinates (flipping X axis)

            # Draw a circle on the index finger to indicate its position
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # Update previous location
            plocX, plocY = clocX, clocY

        # 8. Both index and middle fingers are up : Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find Distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # 10. Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the screen
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display the image
    cv2.imshow("Image", img)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
