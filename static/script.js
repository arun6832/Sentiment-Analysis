document.getElementById('emotion-form').addEventListener('submit', function(event) {
    event.preventDefault();

    let review = document.getElementById('review').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams('review=' + encodeURIComponent(review))
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('log-reg-result').textContent = 'Logistic Regression Prediction: ' + data.log_reg;
    });
});
