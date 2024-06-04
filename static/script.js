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
            console.log(data.message);
        } else {
            console.log(data.message);
        }
    })
    .catch(error => console.error('Error:', error));
});


