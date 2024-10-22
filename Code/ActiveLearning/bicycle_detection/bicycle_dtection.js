var max_images = 20; // max pre-loaded on screen

//update  page progress
eel.expose(data_loading_completed);
function data_loading_completed() {
    console.log("Data loading completed.");
    // 更新前端界面，例如隐藏加载动画
    document.getElementById("loading-status").innerHTML = "Data loading completed.";
}


// 函数：开始训练
async function startTraining() {
    try {
        // 禁用“开始训练”按钮，启用“停止训练”按钮
        document.getElementById('start-training-button').disabled = true;
        document.getElementById('stop-training-button').disabled = false;

        await eel.start_training()();  // 调用后端的 start_training 函数
        console.log("训练已开始。");
    } catch (error) {
        console.error("启动训练时出错：", error);
        // 如果出错，重置按钮状态
        document.getElementById('start-training-button').disabled = false;
        document.getElementById('stop-training-button').disabled = true;
    }
}

// 函数：停止训练
async function stopTraining() {
    try {
        // 启用“开始训练”按钮，禁用“停止训练”按钮
        document.getElementById('start-training-button').disabled = false;
        document.getElementById('stop-training-button').disabled = true;

        await eel.stop_training()();  // 调用后端的 stop_training 函数
        console.log("训练已停止。");
    } catch (error) {
        console.error("停止训练时出错：", error);
        // 如果出错，重置按钮状态
        document.getElementById('start-training-button').disabled = true;
        document.getElementById('stop-training-button').disabled = false;
    }
}

function showError(message) {
    let errorDiv = document.getElementById("error-message");
    errorDiv.style.display = "block";
    errorDiv.textContent = "Error: " + message;
}

// 页面加载完成后通知后端开始数据加载

document.addEventListener('DOMContentLoaded', async function () {
    let loadingStatus = document.getElementById('loading-status');
    // 使用 Promise 处理异步操作
    eel.start_data_loading()()  // 调用异步函数
        .then(() => {
                console.log("Notified backend to start data loading.");
                loadingStatus.innerHTML = "Loading data, please wait";
            })
        .catch(error => {
            // 捕获任何错误并进行处理
            console.error("Error notifying backend to start data loading:", error);
            if (typeof showError === 'function') {
                showError(error.message); // 显示错误消息
            } else {
                console.error("showError is not defined.");
            }
        })
        .finally(() => {
            loadingStatus.style.visibility = 'hidden'; // 确保无论成功还是失败都隐藏加载状态
        });
});


function remove_first_image() {
    try {
        let images = document.getElementById("images").children;
        if (images.length > 0) {
            images[0].remove(); // 移除第一个图像
        }
        let big_image = document.getElementById("big_image");
        big_image.setAttribute("src", ""); // 清空大图像的 src 属性

        // 调用 focus_first_image 并处理返回的 Promise
        return focus_first_image()
            .then(() => {
                // 可在这里处理成功后的逻辑
            })
            .catch(error => {
                // 处理 focus_first_image 中可能抛出的错误
                console.error("Error in remove_first_image:", error);
                if (typeof showError === 'function') {
                    showError(error.message); // 显示错误信息
                } else {
                    console.error("showError is not defined.");
                }
            });
    } catch (error) {
        // 处理同步代码中可能抛出的错误
        console.error("Error in remove_first_image:", error);
        if (typeof showError === 'function') {
            showError(error.message); // 显示错误信息
        } else {
            console.error("showError is not defined.");
        }
    }
}

function focus_first_image() {
    return new Promise((resolve, reject) => {
        try {
            let images = document.getElementById("images").children;
            if (images.length > 0) {
                images[0].setAttribute("class", "image focus_image");
            }
            resolve(); // 如果操作成功，解析 Promise
        } catch (error) {
            console.error("Error in focus_first_image:", error);
            if (typeof showError === 'function') {
                showError(error.message); // 显示错误信息
            } else {
                console.error("showError is not defined.");
            }
            reject(error); // 如果出现错误，拒绝 Promise
        }
    });
}


function add_annotation(is_bicycle) {
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
            // 将异步调用转换为 Promise 处理
            eel.add_annotation(url, is_bicycle)()  // 调用 Python 函数
                .then(() => {
                    return remove_first_image();
                })
                .catch(error => {
                    console.error("Error in add_annotation:", error);
                    if (typeof showError === 'function') {
                        showError(error.message);
                    } else {
                        console.error("showError is not defined.");
                    }
                });
        }
    }
}



function training_loaded() {
    return eel.training_loaded()() // 调用返回 Promise 的异步函数
        .then(result => {
            return result; // 如果成功，返回结果
        })
        .catch(error => {
            // 捕获并处理异步操作中的错误
            console.error("Error in training_loaded:", error);

            // 确保 showError 函数已定义，然后调用它
            if (typeof showError === 'function') {
                showError(error.message); // 显示错误信息
            } else {
                console.error("showError is not defined.");
            }

            return false; // 返回 false 以指示失败
        });
}

function validation_loaded() {
    // 调用 eel.validation_loaded()，它假定返回一个 Promise
    return eel.validation_loaded()()
        .then(result => {
            // Promise 成功时，返回结果
            return result;
        })
        .catch(error => {
            // 捕获并处理 Promise 中的错误
            console.error("Error in validation_loaded:", error);

            // 检查并调用 showError 函数以显示错误信息
            if (typeof showError === 'function') {
                showError(error.message);
            } else {
                console.error("showError is not defined.");
            }

            // 返回 false 表示操作失败
            return false;
        });
}

setInterval(function () {
    // 调用异步函数，并处理返回的 Promise
    eel.estimate_processing_time()()
        .then(processing_time => {
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
        })
        .catch(error => {
            console.error("Error in processing time interval:", error);

            // 确保 showError 函数被定义并调用，以显示错误信息
            if (typeof showError === 'function') {
                showError(error.message);
            } else {
                console.error("showError is not defined.");
            }
        });
}, 10000);

setInterval(function () {
    // 调用异步函数，并处理返回的 Promise
    eel.get_current_accuracies()()
        .then(accuracies => {
            console.log("accuracies: " + accuracies.toString());
            if (accuracies.length > 0) {
                let stats = document.getElementById("stats");
                stats.style.visibility = "visible";
                let fscore = accuracies[0];
                if (fscore > 0) {
                    stats.innerHTML = 'Target Accuracy: F-Score = 0.85 <br />Current Accuracy: F-Score = ' + fscore.toString();
                }
            }
        })
        .catch(error => {
            console.error("Error in accuracy interval:", error);
            if (typeof showError === 'function') {
                showError(error.message);
            } else {
                console.error("showError is not defined.");
            }
        });
}, 5000);

setInterval(function() {
    // 检查新图像用于标注
    validation_loaded()
        .then(validationLoaded => {
            if (!validationLoaded) {
                return false;
            } else {
                document.getElementById("instructions").style.visibility = "visible";
            }
            return training_loaded();
        })
        .then(trainingLoaded => {
            if (!trainingLoaded) {
                // 在这里可能需要插入关于训练模式的消息
            }

            let images = document.getElementById("images");
            let current = images.children.length;

            // 一个递归 Promise 链来处理图像加载
            function loadImages(i) {
                if (i < max_images) {
                    return eel.get_next_image()().then(image_details => {
                        if (image_details == null || image_details.length == 0) {
                            return Promise.resolve();
                        }

                        let image_url = image_details[0];
                        let image_thumbnail = image_details[1];
                        let image_label = image_details[2];

                        let new_image = document.createElement("IMG");
                        new_image.setAttribute("src", image_thumbnail);
                        new_image.setAttribute("alt", image_url);
                        new_image.setAttribute("class", "image");
                        new_image.setAttribute("label", image_label.toString());

                        if (document.getElementById("images").children.length < max_images) {
                            document.getElementById("images").appendChild(new_image);
                        } else {
                            new_image.remove(); // 处理竞争条件
                            return Promise.resolve(); // 结束递归
                        }

                        return loadImages(i + 1); // 加载下一个图像
                    });
                } else {
                    return Promise.resolve(); // 如果达到最大图像数，停止递归
                }
            }

            return loadImages(current); // 开始图像加载
        })
        .then(() => {
            return focus_first_image();
        })
        .catch(error => {
            console.error("Error in image loading interval:", error);
            if (typeof showError === 'function') {
                showError(error.message);
            } else {
                console.error("showError is not defined.");
            }
        });
}, 100);


// LOG ANNOTATIONS
document.addEventListener("keypress", function (event) {
    try {
        console.log(event);
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
