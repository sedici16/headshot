from flask import Flask, render_template, request, send_from_directory
from gradio_client import Client, handle_file 
from PIL import Image
import tempfile
import os
import uuid
import base64
import io

from werkzeug.middleware.proxy_fix import ProxyFix



app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 10 MB
client = Client("theoracle/professional_head", hf_token=os.getenv("HF_TOKEN"))

session_counts = {}

@app.route("/form", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session_id = request.remote_addr
        session_counts[session_id] = session_counts.get(session_id, 0)
        print(f"🔁 New request from {session_id}, current count: {session_counts[session_id]}")


        if session_counts[session_id] >= 20:
            print("❌ Limit reached for session")
            return render_template("form.html", error="❌ Limite massimo di 20 generazioni raggiunto.")
        


        cropped_data = request.form.get("cropped_image")
        if not cropped_data:
            return render_template("form.html", error="❌ Nessuna immagine croppata trovata.")

        print("📤 Cropped image received")

        # Decodifica il base64
        base64_data = cropped_data.split(",")[1]
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Salva su disco
        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join("static/results", filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)
        print(f"✅ Cropped image saved to: {image_path}")




        print("📡 Calling Hugging Face predict API...")
        try:
            result = client.predict(
                image=handle_file(image_path),
                prompt_1=request.form["prompt_1"],
                neg_1=request.form["neg_1"],
                strength_1=float(request.form["strength_1"]),
                guidance_1=int(request.form["guidance_1"]),
                prompt_2=request.form["prompt_2"],
                neg_2=request.form["neg_2"],
                guidance_2=int(request.form["guidance_2"]),
                prompt_3=request.form["prompt_3"],
                neg_3=request.form["neg_3"],
                guidance_3=int(request.form["guidance_3"]),
                api_name="/safe_generate_all_steps"
            )

            print("✅ Hugging Face returned results")

            print("📥 HF response raw:", result)
            print("📐 Type of result:", type(result))

            # Check if result is a tuple and last element is an error string
            if isinstance(result, tuple) and isinstance(result[-1], str) and result[-1].startswith("🛑"):
                print("🚨 Hugging Face returned an error message:")
                print(result[-1])
                return render_template("form.html", error=result[-1])



            for i, r in enumerate(result):
                print(f"🧪 Step {i+1} -> {type(r)} | {r}")


            out_paths = []
            for i, out in enumerate(result[:3]):
                out_file = f"{uuid.uuid4().hex}_step{i+1}.png"
                out_path = os.path.join("static/results", out_file)
                Image.open(out).save(out_path)
                print(f"🖼️ Output image {i+1} saved to: {out_path}")
                out_paths.append(out_path)

            session_counts[session_id] += 1
            print(f"✅ Updated session count: {session_counts}")
            return render_template("form.html", outputs=out_paths, used=session_counts[session_id])

        except Exception as e:
            return render_template("form.html", error=f"❌ {type(e).__name__}: {e}")

    return render_template("form.html")

@app.route('/static/results/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/results', filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
