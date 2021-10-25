var buttonPredict = document.getElementById("predict");
var buttonStop = document.getElementById("stop");

buttonStop.disabled = true;

buttonPredict.onclick = function() {
    // var url = window.location.href + "record_status";
    buttonPredict.disabled = true;
    buttonStop.disabled = false;
    
    // disable download link
    // var downloadLink = document.getElementById("download");
    // downloadLink.text = "";
    // downloadLink.href = "";

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    }
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ record: "true" }));
};

buttonStop.onclick = function() {
    buttonPredict.disabled = false;
    buttonStop.disabled = true;    

    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);

            // enable download link
            // var downloadLink = document.getElementById("download");
            // downloadLink.text = "Download Video";
            // downloadLink.href = "/static/video.avi";
        }
    }
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ record: "false" }));
};

