var vocab = ['character_01_ka','character_02_kha','character_03_ga','character_04_gha','character_05_kna','character_06_cha','character_07_chha','character_08_ja','character_09_jha','character_10_yna','character_11_taamatar','character_12_thaa','character_13_daa','character_14_dhaa','character_15_adna','character_16_tabala','character_17_tha','character_18_da','character_19_dha','character_20_na','character_21_pa','character_22_pha','character_23_ba','character_24_bha','character_25_ma','character_26_yaw','character_27_ra','character_28_la','character_29_waw','character_30_motosaw','character_31_petchiryakha','character_32_patalosaw','character_33_ha','character_34_chhya','character_35_tra','character_36_gya','digit_0','digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9',"unclear"];
var vocab2 = ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह','क्ष','त्र','ज्ञ','०','१','२','३','४','५','६','७','८','९'];

function clearDrawing() {
    var canvas = document.querySelector('#paint');
    var context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
}

function submitDrawing() {
var canvas = document.querySelector('#paint');
imgURI = canvas.toDataURL('image/jpeg', .5)

$.getJSON($SCRIPT_ROOT + '/ocr', {
  imgURI:  imgURI,

}, function(data) {
  var s;
  if(data.result !== "unclear"){
    s = "You entered a: " + vocab2[vocab.indexOf(data.result)]
  }
  else
    s = "The character you entered is unclear"
  $('#result').text( s );
  $('input[name=a]').focus().select();
});
document.getElementById("result").innerHTML = "Working...";
return false;
}

//Here is the main code for the paint window
(function() {
    
    var canvas = document.querySelector('#paint');
    var context = canvas.getContext('2d');
    
    var sketch = document.querySelector('#sketch');
    var sketch_style = getComputedStyle(sketch);
    canvas.width = parseInt(sketch_style.getPropertyValue('width'));
    canvas.height = parseInt(sketch_style.getPropertyValue('height'));
    
    
    // Creating a temp canvas
    var temp_canvas = document.createElement('canvas');
    var temp_context = temp_canvas.getContext('2d');
    temp_canvas.id = 'tmp_canvas';
    temp_canvas.width = canvas.width;
    temp_canvas.height = canvas.height;
    
    sketch.appendChild(temp_canvas);

    var mouse = {x: 0, y: 0};
    var last_mouse = {x: 0, y: 0};
    
    var pencil_points = [];
    
    /* Mouse Capturing Work */
    temp_canvas.addEventListener('mousemove', function(e) {
        mouse.x = typeof e.offsetX !== 'undefined' ? e.offsetX : e.layerX;
        mouse.y = typeof e.offsetY !== 'undefined' ? e.offsetY : e.layerY;
    }, false);
    
    /* Drawing on Paint App */
    temp_context.lineWidth = 15;
    temp_context.lineJoin = 'round';
    temp_context.lineCap = 'round';
    temp_context.strokeStyle = 'blue';
    temp_context.fillStyle = 'blue';
    
    temp_canvas.addEventListener('mousedown', function(e) {
        temp_canvas.addEventListener('mousemove', onPaint, false);
        
        mouse.x = typeof e.offsetX !== 'undefined' ? e.offsetX : e.layerX;
        mouse.y = typeof e.offsetY !== 'undefined' ? e.offsetY : e.layerY;
        
        pencil_points.push({x: mouse.x, y: mouse.y});
        
        onPaint();
    }, false);
    temp_canvas.addEventListener('mouseup', function() {
        temp_canvas.removeEventListener('mousemove', onPaint, false);
        
        // Writing down to real canvas now
        context.drawImage(temp_canvas, 0, 0);
        // Clearing temp canvas
        temp_context.clearRect(0, 0, temp_canvas.width, temp_canvas.height);
        // Emptying up Pencil Points
        pencil_points = [];
    }, false);
    
    var onPaint = function() {
        console.log('onPaint called x:' + mouse.x + ', y:' + mouse.y);
        // Saving all the points in an array
        pencil_points.push({x: mouse.x, y: mouse.y});
        
        if (pencil_points.length < 3) {
            var b = pencil_points[0];
            temp_context.beginPath();
            temp_context.arc(b.x, b.y, temp_context.lineWidth / 2, 0, Math.PI * 2, !0);
            temp_context.fill();
            temp_context.closePath();
            return;
        }
        
        temp_context.clearRect(0, 0, temp_canvas.width, temp_canvas.height);
        temp_context.beginPath();
        temp_context.moveTo(pencil_points[0].x, pencil_points[0].y);
        
        for (var i = 1; i < pencil_points.length - 2; i++) {
            var c = (pencil_points[i].x + pencil_points[i + 1].x) / 2;
            var d = (pencil_points[i].y + pencil_points[i + 1].y) / 2;
            temp_context.quadraticCurveTo(pencil_points[i].x, pencil_points[i].y, c, d);
        }
        // For the last 2 points
        temp_context.quadraticCurveTo(
            pencil_points[i].x,
            pencil_points[i].y,
            pencil_points[i + 1].x,
            pencil_points[i + 1].y
        );
        temp_context.stroke();
    };
}());