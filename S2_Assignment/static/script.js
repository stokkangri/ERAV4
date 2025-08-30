document.addEventListener('DOMContentLoaded', function() {
    // Animal selection functionality
    const animalOptions = document.querySelectorAll('input[name="animal"]');
    const animalDisplay = document.getElementById('animalDisplay');
    const animalImage = document.getElementById('animalImage');
    const animalName = document.getElementById('animalName');

    animalOptions.forEach(option => {
        option.addEventListener('change', function() {
            if (this.checked) {
                const selectedAnimal = this.value;
                showAnimal(selectedAnimal);
            }
        });
    });

    function showAnimal(animal) {
        fetch(`/animal/${animal}`)
            .then(response => response.json())
            .then(data => {
                if (data.image_path) {
                    animalImage.src = data.image_path;
                    animalName.textContent = data.animal;
                    animalDisplay.style.display = 'block';
                    
                    // Add animation
                    animalDisplay.style.opacity = '0';
                    animalDisplay.style.transform = 'scale(0.8)';
                    setTimeout(() => {
                        animalDisplay.style.transition = 'all 0.5s ease';
                        animalDisplay.style.opacity = '1';
                        animalDisplay.style.transform = 'scale(1)';
                    }, 10);
                    
                    // Handle image load success
                    animalImage.onload = function() {
                        animalImage.style.display = 'block';
                        animalName.textContent = data.animal;
                    };
                    
                    // Handle image load error with fallback
                    animalImage.onerror = function() {
                        console.log(`Image failed to load: ${data.image_path}, trying .jpeg extension...`);
                        // Try .jpeg extension as fallback
                        const jpegPath = data.image_path.replace('.jpg', '.jpeg');
                        animalImage.src = jpegPath;
                        
                        // If .jpeg also fails, show emoji fallback
                        animalImage.onerror = function() {
                            animalImage.style.display = 'none';
                            animalName.textContent = `${animal} (Image not available)`;
                        };
                    };
                }
            })
            .catch(error => {
                console.error('Error fetching animal:', error);
                // Fallback to emoji if image fails to load
                animalImage.style.display = 'none';
                animalName.textContent = `${animal} (Image not available)`;
                animalDisplay.style.display = 'block';
            });
    }

    // File upload functionality
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileType = document.getElementById('fileType');

    // File input change event
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Click to upload - improved click handling
    uploadArea.addEventListener('click', (e) => {
        // Don't trigger if clicking on the button
        if (e.target.classList.contains('browse-btn')) {
            return;
        }
        console.log('Upload area clicked, triggering file input...');
        fileInput.click();
    });
    
    // Also handle button click separately
    document.querySelector('.browse-btn').addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent upload area click
        console.log('Browse button clicked, triggering file input...');
        fileInput.click();
    });

    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        console.log('File input change event triggered');
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name, 'Size:', file.size, 'Type:', file.type);
            handleFile(file);
        } else {
            console.log('No file selected');
        }
    }

    function handleFile(file) {
        console.log('Handling file upload for:', file.name);
        
        // Show loading state
        uploadArea.classList.add('loading');
        
        const formData = new FormData();
        formData.append('file', file);
        
        console.log('FormData created, sending to server...');

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Server response received:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('Server data:', data);
            uploadArea.classList.remove('loading');
            
            if (data.error) {
                console.error('Server error:', data.error);
                showError(data.error);
            } else {
                console.log('File upload successful, displaying info...');
                displayFileInfo(data);
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            uploadArea.classList.remove('loading');
            showError('Failed to upload file. Please try again.');
        });
    }

    function displayFileInfo(data) {
        fileName.textContent = data.filename;
        fileSize.textContent = formatFileSize(data.filesize);
        fileType.textContent = data.filetype;
        
        // Show file info with animation
        fileInfo.style.display = 'block';
        fileInfo.style.opacity = '0';
        fileInfo.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            fileInfo.style.transition = 'all 0.5s ease';
            fileInfo.style.opacity = '1';
            fileInfo.style.transform = 'translateY(0)';
        }, 10);

        // Show success message
        showSuccess('File uploaded successfully!');
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function showError(message) {
        showNotification(message, 'error');
    }

    function showSuccess(message) {
        showNotification(message, 'success');
    }

    function showNotification(message, type) {
        // Remove existing notifications
        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        `;

        if (type === 'error') {
            notification.style.background = '#e74c3c';
        } else {
            notification.style.background = '#27ae60';
        }

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }, 5000);
    }

    // Add some interactive effects
    document.addEventListener('mousemove', function(e) {
        const boxes = document.querySelectorAll('.box');
        boxes.forEach(box => {
            const rect = box.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            box.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
        });
    });

    document.addEventListener('mouseleave', function() {
        const boxes = document.querySelectorAll('.box');
        boxes.forEach(box => {
            box.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)';
        });
    });
});
