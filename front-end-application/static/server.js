function predict(btn) {
    console.log(btn)
    btn.disabled = true;
    
    // XMLHttpRequest
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ record: "true" }));
}

var instr_vid = document.getElementById("instructions");

function play_instr(btn) {
    instr_vid.play();
    btn.disabled = true;
}
  
instr_vid.addEventListener('ended', function(e){
    var btn = document.getElementById("instr_button");
    btn.disabled = false;
    btn.blur();
});