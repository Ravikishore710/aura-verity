# AuraVerity: A Hybrid Deepfake Detection dApp

**Tagline:** *Bringing trust and transparency to digital media with decentralized verification.*

---

### Problem Statement

The rise of sophisticated deepfake technology poses a significant threat to the integrity of digital media, making it difficult to distinguish between real and manipulated content. This erodes trust in online information and can be used for malicious purposes. AuraVerity aims to combat this by providing an accessible, powerful, and transparent tool for deepfake detection.

---

### Key Features

* **Advanced AI Detection:** Utilizes a PyTorch-based model to provide a "REAL" or "FAKE" verdict on uploaded images and videos.
* **Forensic Analysis:** Provides a suite of supporting forensic checks (Metadata, Facial Geometry, Skin Texture) to give users deeper insights.
* **Decentralized Frontend:** The user interface is hosted on the Internet Computer (ICP), making it secure, fast, and censorship-resistant.
* **Hybrid Architecture:** Combines the security of a decentralized frontend with the raw power of a centralized Python backend for heavy AI computation.
* **Visual Heatmaps:** Generates a visual representation of the areas the AI model focuses on during its analysis.

---

### Tech Stack

* **AI Backend:** Python, Flask, PyTorch, dlib, OpenCV
* **Decentralized Frontend:** Internet Computer (ICP), Motoko, HTML, Tailwind CSS, JavaScript
* **Development Environment:** Windows Subsystem for Linux (WSL) for ICP, Windows for the Python backend.

---

### System Architecture

The project uses a hybrid model where the user-facing application is decentralized, but the computationally intensive AI processing is handled by a traditional server.

```
[User's Browser] <--> [ICP Frontend Canister (truth_chain)] <--> [Flask API Server (deepfake_webapp)]
     |                                                                     |
     |                                                               [PyTorch Model]
     |                                                                     |
     +------------------------------------------------------------> [Analysis Results]
```

---

### How to Run the Application

To run the full application, you will need **three separate terminals open simultaneously.**

**Terminal 1: Start the ICP Local Replica (Ubuntu)**
```bash
cd truth_chain
dfx start --clean
```

**Terminal 2: Start the Python Backend Server (Windows CMD)**
```cmd
cd deepfake_webapp
py -3.11 app.py
```

**Terminal 3: Deploy Canisters (Ubuntu)**
```bash
cd truth_chain
dfx deploy
```

Once deployed, access the frontend canister URL provided in the terminal.

---

### Screenshots


<img width="1914" height="911" alt="image" src="https://github.com/user-attachments/assets/b16b96a5-15fd-45ce-be07-bc2714cb727a" />

<img width="1898" height="900" alt="image" src="https://github.com/user-attachments/assets/1479945d-3b1e-448e-957e-1ff8fea41607" />

<img width="728" height="411" alt="image" src="https://github.com/user-attachments/assets/65efd3f7-fe3a-4678-bbc3-2cad9b1a86ae" />



---

### Future Improvements

* **On-Chain Quota System:** Implement a canister-based system to limit user uploads per day.
* **Immutable Logs:** Store all analysis results in a dedicated, tamper-proof history canister.
* **Token-Based Rewards:** Introduce a utility token to reward users for contributing to the detection ecosystem.

---

### Team

* **Ravi Kishore:** Lead Developer & AI Engineer
* **Riya Verma:** AI Developer
