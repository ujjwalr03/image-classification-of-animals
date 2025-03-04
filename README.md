## **📌 Project Description: Image Classification of Animals**  

This project builds a **Deep Learning model** using **MobileNetV2 (Transfer Learning)** to classify images of animals into **15 categories**. It provides **two execution options**:  

---

## **🛠 Features**  
✅ **Uses Pre-trained MobileNetV2 for Fast & Accurate Classification**  
✅ **15 Animal Classes (Bear, Dog, Cat, Lion, Tiger, etc.)**  
✅ **Trains & Saves Model for Future Predictions**  
✅ **Supports User-Uploaded Image Testing**  
✅ **No Accuracy Graphs – Only Image Detection Output**  

---

## **🔹 Technologies Used**
- **TensorFlow / Keras** → Deep Learning framework  
- **MobileNetV2** → Transfer Learning for fast & accurate classification  
- **OpenCV** → Image preprocessing  

---

## **💻 Steps to Run Locally**
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ujjwalr03/image-classification-of-animals.git
   cd image-classification-of-animals
   ```
2. **Install Dependencies**  
   ```bash
   pip install tensorflow opencv-python numpy
   ```
3. **Ensure Your Dataset is in `dataset/` Folder**  
   ```
   /dataset
   │── /Bear
   │── /Bird
   │── /Cat
   │── /Dog
   │── /Elephant
   │── ... (Other Classes)
   ```
4. **Run the script**  
   ```bash
   python image-classification-of-animals.py
   ```
5. **Test an Image**
   - Place test images inside `test_images/`
   - Modify the script to use the correct image path  
   ```bash
   python image-classification-of-animals.py
   ```

---

## **🔗 Open in Google Colab**
Click below to open the notebook in Google Colab:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ujjwalr03/image-classification-of-animals/blob/main/image-classification-of-animals-colab.ipynb)  

### **Steps in Google Colab**
1. Click the **"Open in Colab"** button above.  
2. Run all cells (dataset is downloaded automatically).  
3. Upload an image when prompted for testing.  

---

## 📜 **License**
This project is open-source under the **MIT License**.  

