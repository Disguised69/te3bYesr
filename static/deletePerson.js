  document.getElementById('deleteButton').addEventListener('click', function() {
        const name = document.getElementById('nameToDelete').value;

        if (name) {
            fetch(`/api/v1/delete_person/${name}`, {
                method: 'DELETE'
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert(data.message);
                }
            })
            .catch(error => console.error('Error:', error));
        } else {
            alert('Please enter a name to delete.');
        }
    });