let name = '';

document.getElementById("confirmButton").addEventListener("click", function (message) {
    name = document.getElementById("name").value.trim();
    if (name) {
        var formData = new FormData();
        formData.append("name", name);

        fetch("/api/v1/newPerson", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (response.ok) {
                console.log("Folder created successfully");
                alert("folder for "+ name+" created sucessfully")
                // Optionally, perform additional actions after successful folder creation
            } else {
                console.error("Error creating folder:", response.statusText);
                // Optionally, handle error case
                alert("Person already exists")
            }
        })
        .catch(error => {
            console.error("Error creating folder:", error);
            // Optionally, handle error case
        });
    } else {
        console.error("Person name is required");
        alert("Person name is required");
        // Optionally, display error message to the user
    }
});

document.getElementById("takePictureButton").addEventListener("click", function () {
    if (!name) {
        alert("Please enter a name and confirm first.");
        return;
    }

fetch('/capture', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({ name: name })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            console.log(`Total pictures taken: ${data.count}`);
            const countElement = document.getElementById("pictureCount");
            countElement.textContent = `Total pictures taken: ${data.count}`;
        } else {
            alert(data.message);
        }
    })
    .catch(error => alert('Error: no face detected'));
});
 async function resumeFaceRecognition() {
            try {
                const response = await fetch('/resume_capture', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });

                if (!response.ok) {
                    throw new Error('Failed to resume face recognition');
                }

                console.log('Face recognition resumed successfully.');
            } catch (error) {
                console.error('Error resuming face recognition:', error.message);
            }
        }

document.getElementById("back").addEventListener("click", function (){
            resumeFaceRecognition();
            window.location.href = "/main";
        });