 const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        const errorDiv = document.getElementById('error');

        let selectedFile = null;

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleFile(file);
            }
        });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                resultContainer.style.display = 'none';
                errorDiv.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        predictBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('file', selectedFile);

            predictBtn.disabled = true;
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            errorDiv.style.display = 'none';

            try {
                const response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.prediction) {
                    resultText.textContent = data.prediction;
                    resultContainer.style.display = 'block';
                } else if (data.error) {
                    errorDiv.textContent = 'Error: ' + data.error;
                    errorDiv.style.display = 'block';
                }
            } catch (err) {
                errorDiv.textContent = 'Failed to connect to the server. Make sure the backend is running on http://localhost:8000';
                errorDiv.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                predictBtn.disabled = false;
            }
        });