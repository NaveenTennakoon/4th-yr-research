function predict(btn) {
    console.log(btn)
    btn.disabled = true;
    
    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ record: "true" }));
}