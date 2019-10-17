var train_xor = true;
// var train_or = true;

function setup(){
    createCanvas(500,500);
    background(0);

    nn = new RedeNeural(2,3,1);
    

    // XOR Problem
    dataset1 = {
        inputs:
        [
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0]
        ],
        outputs:
        [
            [0],
            [1],
            [1],
            [0]
        ]
    }
    // // OR Problem
    // dataset2 = {
    //     inputs:
    //     [
    //         [1, 1],
    //         [1, 0],
    //         [0, 1],
    //         [0, 0]
    //     ],
    //     outputs:
    //     [
    //         [1],
    //         [1],
    //         [1],
    //         [0]
    //     ]
    // }
}


function draw(){
    if(train_xor){
        for(var i=0; i<10000; i++){
            var index = floor(random(4));
            nn.train(dataset1.inputs[index], dataset1.outputs[index]);
        }
        if(nn.predict_xor([0, 0])[0] < 0.01 && nn.predict_xor([1, 0])[0] > 0.99 && nn.predict_xor([1, 1])[0] < 0.01 && nn.predict_xor([0, 1])[0] > 0.99){
            console.log("Terminou a XOR");
            train_xor = false;
        }
    }
    
    // if(train_or){
    //     for(var i=0; i<10000; i++){
    //         var index = floor(random(4));
    //         nn.train(dataset2.inputs[index], dataset2.outputs[index]);
    //     }
    //     if(nn.predict_or([0, 0])[0] < 0.01 && nn.predict_or([1, 0])[0] > 0.99 && nn.predict_or([1, 1])[0] > 0.99 && nn.predict_or([0, 1])[0] > 0.99){
    //         console.log("Terminou a OR");
    //         train_or = false;
    //     }
    // }
}