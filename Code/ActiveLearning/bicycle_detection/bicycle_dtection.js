var max_images = 20; // max pre-loaded on screen


//update  page progress
eel.expose(data_loading_completed);
function data_loading_completed() {
    console.log("Data loading completed.");
    // 您可以在此更新前端界面，例如隐藏加载动画
    document.getElementById("loading-status").style.visibility = "hidden";  // 隐藏加载状态
}

eel.expose(data_annotation_completed);
function data_annotation_completed() {
    console.log("Data annotation completed.");
    // 您可以在此更新前端界面，例如隐藏加载动画
    //document.getElementById("annotation-status").style.visibility = "hidden";  
    document.getElementById("annotation-status").innerHTML = "Data annotation completed";
    // 隐藏加载状态
}



function showError(message) {
    let errorDiv = document.getElementById("error-message");
    errorDiv.style.display = "block";
    errorDiv.textContent = "Error: " + message;
}

// 页面加载完成后通知后端开始数据加载

document.addEventListener('DOMContentLoaded', async function () {
    let loadingStatus = document.getElementById('loading-status');
    try {
        loadingStatus.style.visibility = 'visible';
        await eel.start_data_loading()();
        console.log("Notified backend to start data loading.");
        loadingStatus.style.visibility = 'hidden'; 
    } catch (error) {
        console.error("Error notifying backend to start data loading:", error);
        showError(error.message);
        loadingStatus.style.visibility = 'hidden';

    }
});



async function remove_first_image() {
    try {
        let images = document.getElementById("images").children;
        if (images.length > 0) {
            images[0].remove();
        }
        let big_image = document.getElementById("big_image");
        big_image.setAttribute("src", "");

        await focus_first_image();
    } catch (error) {
        console.error("Error in remove_first_image:", error);
        showError(error.message);
    }
}

async function focus_first_image() {
    try {
        let images = document.getElementById("images").children;
        if (images.length > 0) {
            images[0].setAttribute("class", "image focus_image");
        }
    } catch (error) {
        console.error("Error in focus_first_image:", error);
        showError(error.message);
    }
}

async function add_annotation(is_bicycle) {
    try {
        let images = document.getElementById("images").children;
        if (images.length > 0) {
            let url = images[0].alt;
            let label = images[0].getAttribute("label");

            let annotation_ok = true;
            if (label !== null && label !== undefined && label !== "") {
                if (label.toString() == "0" && is_bicycle) {
                    alert("Warning: this image does *not* contain a bicycle, according to existing annotations");
                    annotation_ok = false;
                } else if (label.toString() == "1" && !is_bicycle) {
                    alert("Warning: this image *does* contain a bicycle, according to existing annotations");
                    annotation_ok = false;
                }
            }
            if (annotation_ok) {
                await eel.add_annotation(url, is_bicycle)(); // Call to python function
                await remove_first_image();
            }
        }
    } catch (error) {
        console.error("Error in add_annotation:", error);
        showError(error.message);
    }
}

async function training_loaded() {
    try {
        return await eel.training_loaded()();
    } catch (error) {
        console.error("Error in training_loaded:", error);
        showError(error.message);
        return false;
    }
}

async function validation_loaded() {
    try {
        return await eel.validation_loaded()();
    } catch (error) {
        console.error("Error in validation_loaded:", error);
        showError(error.message);
        return false;
    }
}

//better comment
/*
setInterval(async function () {
    try {
        // Check for updated accuracy scores every 10 seconds
        let processing_time = await eel.estimate_processing_time()();
        let time_div = document.getElementById("time");

        console.log("processing time: " + processing_time.toString());

        if (processing_time > 0) {
            let message = "";
            if (processing_time < 90) {
                message = Math.floor(processing_time).toString() + " seconds ";
            } else if (processing_time < 240) {
                message = (Math.round(processing_time / 30) / 2).toString() + " minutes ";
            } else if (processing_time < 600) {
                message = Math.round(processing_time / 60).toString() + " minutes ... maybe get a cup of coffee ";
            } else if (processing_time < 5400) {
                message = Math.round(processing_time / 60).toString() + " minutes ... maybe take a short break and get some exercise ";
            } else {
                message = (Math.round(processing_time / 1800) / 2).toString() + " hours ... maybe have a meal and come back later ";
            }

            time_div.style.visibility = "visible";
            time_div.innerHTML = '<b>Estimated Time remaining to prepare annotated images for machine learning (download and extract COCO and ImageNet vectors):</b><br /> ' + message;
        } else {
            time_div.style.visibility = "hidden";
        }
    } catch (error) {
        console.error("Error in processing time interval:", error);
        showError(error.message);
    }
}, 10000);
*/

setInterval(async function () {
    try {
        // Check for updated accuracy scores every 5 seconds
        let accuracies = await eel.get_current_accuracies()();
        console.log("accuracies: " + accuracies.toString());
        if (accuracies.length > 0) {
            let stats = document.getElementById("stats");
            stats.style.visibility = "visible";
            let fscore = accuracies[0];
            if (fscore > 0) {
                stats.innerHTML = 'Target Accuracy: F-Score = 0.85 <br />Current Accuracy: F-Score = ' + fscore.toString();
            }
        }
    } catch (error) {
        console.error("Error in accuracy interval:", error);
        showError(error.message);
    }
}, 5000);

setInterval(async function(){ 
    //check for new images to annotate every half second    
    try{
    if(!validation_loaded()){        
        return false;
    }
    else{
        document.getElementById("instructions").style="visibility:visible";
    }
    if(!training_loaded){
        // TODO: MESSAGE ABOUT PRACTICE MODE?
    }

    images = document.getElementById("images");

    current = images.children.length;

    for(var i = current; i <= max_images; i++){
        image_details = await eel.get_next_image()(); // Call to python function
        if(image_details == null || image_details.length == 0){
            break;
        }
        image_url = image_details[0];
        image_thumbnail = image_details[1];
        image_label = image_details[2];
        
        new_image = document.createElement("IMG");
        new_image.setAttribute("src", image_thumbnail);
        new_image.setAttribute("alt", image_url);
        new_image.setAttribute("class", "image");
        new_image.setAttribute("label", image_label.toString());
         
        if(document.getElementById("images").children.length <= max_images){
            document.getElementById("images").appendChild(new_image);
        }
        else{
            new_image.remove(); // race condition: we are already ok
            break;
        }
    }    
    focus_first_image(); 
    } catch (error) {
        console.error("Error in image loading interval:", error);
        showError(error.message);
    }
}, 500);

// LOG ANNOTATIONS
document.addEventListener("keypress", function (event) {
    try {
        // console.log(event);
        if (event.key === "b") {
            add_annotation(true);
        } else if (event.key === "n") {
            add_annotation(false);
        } else if (event.key === "z") {
            let images = document.getElementById("images").children;
            if (images.length > 0) {
                let big_image = document.getElementById("big_image");
                let url = images[0].alt;
                big_image.setAttribute("src", url);
            } else {
                alert("No image available to zoom.");
            }
        }
    } catch (error) {
        console.error("Error in keypress event handler:", error);
        showError(error.message);
    }
});
