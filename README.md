# 📝 Movie Genre Prediction using Multimodal Learning

👩‍💻 **Author: Shaun Job**

---

## 🎯 Motivation – Why did I pick this topic?

I've always been fascinated by how humans process information from multiple sources — what we see 👀, hear 👂, read 📖 — to make decisions. Inspired by this, I wanted to explore **multimodal learning**, where machine learning models combine data from different modalities.

Specifically, I chose **movie genre prediction** because:

✅ It combines two rich modalities:

- **Textual data** (movie descriptions)
- **Visual data** (movie posters)

✅ It’s highly practical: such models could assist **content recommendation systems** like Netflix, IMDb, or streaming services.

✅ I love movies 🎬 — and working with movie data sounded fun and relatable!

---

## 🔍 Connection with past and current work in Multimodal Learning

Multimodal learning has rapidly grown in recent years. Early machine learning models primarily worked with **single modalities** (e.g., text or images). However, researchers soon realized that **combining modalities improves performance** in many tasks — just like humans!

Some key milestones:

📝 **2015-2017**: Models like **VQA (Visual Question Answering)** emerged, combining images + text to answer questions.

🤖 **2019-2021**: Pretrained multimodal transformers like **CLIP (OpenAI)** and **VisualBERT** allowed learning joint embeddings of images and text at scale.

🚀 **Today**: Multimodal models power **image captioning**, **text-to-image generation** (e.g., DALL·E, Midjourney), **video understanding**, and more.

My project is a **simpler multimodal fusion** — concatenating sentence embeddings + image features to train a genre classifier. It reflects an **applied, lightweight multimodal pipeline** (without large transformers).

---

## 🧠 What did I learn from this project?

Here’s what I learned through building this pipeline:

✅ **Feature extraction matters a lot!**

- I used **sentence-transformers (all-MiniLM-L6-v2)** for text encoding → very efficient and surprisingly strong embeddings.
- For image features, I used a pretrained **MobileNetV2** → lightweight yet effective.

✅ **Data quality issues are real**

- Some movies had missing or broken poster URLs → I had to handle missing images by inserting zero-vectors to avoid breaking the pipeline.

✅ **Multilabel classification is trickier than single-label**

- A movie can belong to **multiple genres** → requires using **OneVsRestClassifier** rather than standard multiclass classifiers.
- Evaluation needs metrics like **micro/macro F1**, **multilabel confusion matrices**.

✅ **Combining features improves performance**

- Using **both text + image features performed better** than text-only or image-only (from earlier experiments I tried).

---

## 💻 Code, Experiments & Visualizations

👉 The complete implementation can be found in the **attached code notebook**.

Key components:

✅ Downloading posters from URLs → saved locally.

✅ Encoding descriptions using **sentence_transformers**.

✅ Extracting poster features with **torchvision.models.mobilenet_v2**.

✅ Concatenating text + image features into one feature matrix.

✅ Training **LightGBM classifier** in **One-vs-Rest** setup.

✅ Evaluating using **accuracy, precision, recall, F1, confusion matrix**.

---

## Sample output visualization (from my notebook):

### 📊 Classification Report

| Genre             | Precision | Recall | F1-Score | Support |
|------------------|:---------:|:------:|:--------:|:-------:|
| Action            | 0.76      | 0.28   | 0.41     | 232     |
| Adventure         | 0.78      | 0.38   | 0.51     | 239     |
| Animation         | 0.94      | 0.38   | 0.54     | 80      |
| Comedy            | 0.85      | 0.32   | 0.47     | 298     |
| Crime             | 0.85      | 0.33   | 0.47     | 218     |
| Documentary       | 1.00      | 0.29   | 0.44     | 7       |
| Drama             | 0.95      | 0.21   | 0.35     | 434     |
| Family            | 0.92      | 0.53   | 0.67     | 164     |
| Fantasy           | 0.91      | 0.41   | 0.57     | 207     |
| History           | 0.95      | 0.26   | 0.41     | 81      |
| Horror            | 0.93      | 0.35   | 0.51     | 109     |
| Music             | 0.82      | 0.28   | 0.42     | 32      |
| Mystery           | 0.95      | 0.37   | 0.53     | 155     |
| Romance           | 0.80      | 0.33   | 0.46     | 232     |
| Science Fiction   | 0.90      | 0.35   | 0.50     | 188     |
| TV Movie          | 1.00      | 0.12   | 0.21     | 34      |
| Thriller          | 0.76      | 0.39   | 0.51     | 400     |
| War               | 0.89      | 0.37   | 0.52     | 43      |
| Western           | 0.75      | 0.33   | 0.46     | 9       |
| Action (2)        | 0.85      | 0.39   | 0.53     | 273     |
| Adventure (2)     | 1.00      | 0.33   | 0.50     | 88      |
| Animation (2)     | 0.81      | 0.55   | 0.65     | 161     |
| Comedy (2)        | 0.86      | 0.43   | 0.57     | 294     |
| Crime (2)         | 0.94      | 0.43   | 0.59     | 69      |
| Documentary (2)   | 1.00      | 0.38   | 0.55     | 32      |
| Drama (2)         | 0.82      | 0.34   | 0.48     | 373     |
| Family (2)        | 0.83      | 0.19   | 0.31     | 53      |
| Fantasy (2)       | 1.00      | 0.35   | 0.52     | 51      |
| History (2)       | 1.00      | 0.57   | 0.73     | 7       |
| Horror (2)        | 0.90      | 0.43   | 0.58     | 153     |
| Music (2)         | 1.00      | 0.31   | 0.47     | 13      |
| Mystery (2)       | 1.00      | 0.30   | 0.46     | 27      |
| Romance (2)       | 0.95      | 0.28   | 0.43     | 68      |
| Science Fiction (2)| 0.96     | 0.37   | 0.53     | 65      |
| TV Movie (2)      | 0.50      | 0.50   | 0.50     | 4       |
| Thriller (2)      | 1.00      | 0.34   | 0.51     | 105     |
| War (2)           | 1.00      | 0.25   | 0.40     | 8       |
| Western (2)       | 1.00      | 0.26   | 0.42     | 19      |

| Metric           | Precision | Recall | F1-Score | Support |
|-----------------|:---------:|:------:|:--------:|:-------:|
| Micro Avg        | 0.86      | 0.35   | 0.50     | 5025    |
| Macro Avg        | 0.90      | 0.35   | 0.49     | 5025    |
| Weighted Avg     | 0.87      | 0.35   | 0.50     | 5025    |
| Samples Avg      | 0.45      | 0.35   | 0.38     | 5025    |

suffix(2) is used for duplicate genre names (likely from multilabel setup or multiple genre tags).
### Testing Accuracy

![Accuracy report](https://github.com/user-attachments/assets/9b44e5e8-5e9f-462a-960a-4248efa13a99)

### Confusion Matrices for all Genre's

![Confusion_1](https://github.com/user-attachments/assets/407ae517-ac5c-441f-9148-1ab4458615e3)

![confusion_2](https://github.com/user-attachments/assets/869e3271-84a7-4dd0-b764-0c8af42002fc)

![Confusion_3](https://github.com/user-attachments/assets/3f517a42-d598-450e-b170-a458dccee272)

![Confusion_4](https://github.com/user-attachments/assets/14337d39-bea3-4145-bdaf-febe02c2d93a)

![confusion_5](https://github.com/user-attachments/assets/bf9ea9ab-c3c1-422b-82e8-6a684e2863e9)

![confusion_6](https://github.com/user-attachments/assets/583d9a93-45a5-49f4-bff3-66bbba39b464)

![Confusion_7](https://github.com/user-attachments/assets/1a8cc46d-f682-4fcf-88ee-6b3c58aa29d0)


---

## 🤔 Reflections

### (a) What surprised me?

😮 I was surprised by how **strong textual features alone were!**  
Even without image features, the description text often carried enough signal for genre prediction.

Also, I expected image features to be more predictive → but **movie posters aren’t always representative** of the genre (e.g., minimalistic posters, artistic posters). They’re more marketing artifacts than descriptive visual cues.

Finally, I didn’t anticipate how much **genre overlap** exists → some genres frequently co-occur (e.g., Action + Adventure, Drama + Romance), which complicates classification.

---

### (b) Scope for improvement?

🚩 **Better image models:** I used MobileNetV2 for efficiency. Using a more powerful vision backbone like **ResNet-50** or a multimodal pretrained model like **CLIP** might yield stronger image features.

🚩 **Late fusion / attention-based fusion:** Instead of simply concatenating features, more advanced techniques (e.g., attention-based fusion, multimodal transformers) could better model interactions.

🚩 **More metadata:** We could include additional modalities → cast info, director, release year, user reviews → to make it truly multimodal.

🚩 **Fine-tuning transformers:** Instead of using pretrained frozen encoders, fine-tuning on a movie-specific dataset could further improve embeddings.

---

## 📚 References

Here are the resources/tools I used throughout the project:

- **sentence-transformers library:** [https://www.sbert.net/](https://www.sbert.net/)
- **MobileNetV2 paper:** [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- **CLIP:** [https://openai.com/research/clip](https://openai.com/research/clip)
- **LightGBM:** [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
- **Multimodal learning overview:** [https://distill.pub/2020/multimodal/](https://distill.pub/2020/multimodal/)
- Example technical blogs:
  - [https://jalammar.github.io/](https://jalammar.github.io/)
  - [https://colah.github.io/](https://colah.github.io/)
  - [https://bair.berkeley.edu/blog/2024/07/20/visual-haystacks/](https://bair.berkeley.edu/blog/2024/07/20/visual-haystacks/)
- **Class notes from DA623 (Winter 2025)**

---

## 🙌 Final thoughts

This project gave me **hands-on experience building an end-to-end multimodal learning pipeline.**  
It helped bridge theory (multimodal embeddings, classifier design) and practice (handling missing data, evaluating multilabel outputs).

💡 In future work, I’m excited to explore **multimodal transformers** or even try **generative multimodal tasks!**

---

Thanks for reading! 😊

👉 You can find the **complete code notebook and outputs in this repository.**
