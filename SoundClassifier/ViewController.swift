//
//  ViewController.swift
//  SoundClassifier
//
//  Created by M'haimdat omar on 24-08-2019.
//  Copyright © 2019 M'haimdat omar. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController, AVCaptureAudioDataOutputSampleBufferDelegate {
    
    /////////////////////////////
    // MARK: - Class's attributes
    lazy var label = UILabel()
    private enum classes: String {
        case dog = "dog"
        case rooster = "rooster"
        case helicopter = "helicopter"
        case sneezing = "sneezing"
    }
    
    lazy var output = ["dog", "rooster", "helicopter", "sneezing"]

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = #colorLiteral(red: 0.9999960065, green: 1, blue: 1, alpha: 1)
        self.label.text = "No matching"
        setupLabel()
        getInference()
    }
    
    ///////////////////////////////////////////////////////////
    // MARK: - Setup the label layout and add it to the subview
    private func setupLabel() {
        label.translatesAutoresizingMaskIntoConstraints = false
        label.font = UIFont(name: "Avenir-Heavy", size: 70)
        view.addSubview(label)
        label.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        label.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
    }
    
    
    /////////////////////////////////////////////////////////////////////////////
    // MARK: - Get the inference for each sound and change the layout accordingly
    private func getInference() {
        //200 layers 100 iterations
        let model = SoundClassification()
        
        var count = 0
        _ = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { t in
            
            var wav_file: AVAudioFile!
            do {
                let fileUrl = URL(fileReferenceLiteralResourceName: "\(self.output[count])_1_converted.wav")
                wav_file = try AVAudioFile(forReading:fileUrl)
            } catch {
                fatalError("Could not open wav file.")
            }
            
            print("wav file length: \(wav_file.length)")
            assert(wav_file.fileFormat.sampleRate==16000.0, "Sample rate is not right!")
            
            let buffer = AVAudioPCMBuffer(pcmFormat: wav_file.processingFormat,
                                          frameCapacity: UInt32(wav_file.length))
            do {
                try wav_file.read(into:buffer!)
            } catch{
                fatalError("Error reading buffer.")
            }
            guard let bufferData = buffer?.floatChannelData else { return }
            
            // Chunk data and set to CoreML model
            let windowSize = 15600
            guard let audioData = try? MLMultiArray(shape:[windowSize as NSNumber],
                                                    dataType:MLMultiArrayDataType.float32)
                else {
                    fatalError("Can not create MLMultiArray")
            }
            
            // Ignore any partial window at the end.
            var results = [Dictionary<String, Double>]()
            let windowNumber = wav_file.length / Int64(windowSize)
            for windowIndex in 0..<Int(windowNumber) {
                let offset = windowIndex * windowSize
                for i in 0...windowSize {
                    audioData[i] = NSNumber.init(value: bufferData[0][offset + i])
                }
                let modelInput = SoundClassificationInput(audio: audioData)
                
                guard let modelOutput = try? model.prediction(input: modelInput) else {
                    fatalError("Error calling predict")
                }
                results.append(modelOutput.categoryProbability)
            }
            
            var prob_sums = Dictionary<String, Double>()
            for r in results {
                for (label, prob) in r {
                    prob_sums[label] = (prob_sums[label] ?? 0) + prob
                }
            }
            
            var max_sum = 0.0
            var max_sum_label = ""
            for (label, sum) in prob_sums {
                if sum > max_sum {
                    max_sum = sum
                    max_sum_label = label
                }
            }
            
            let most_probable_label = max_sum_label
            let probability = max_sum / Double(results.count)
            print("\(most_probable_label) predicted, with probability: \(probability)")
            
            let prediction: classes = classes.init(rawValue: most_probable_label)!
            
            switch prediction {
            case .dog:
                self.label.text = most_probable_label
                self.view.backgroundColor = #colorLiteral(red: 0.2392156869, green: 0.6745098233, blue: 0.9686274529, alpha: 1)
            case .helicopter:
                self.label.text = most_probable_label
                self.view.backgroundColor = #colorLiteral(red: 0.8078431487, green: 0.02745098062, blue: 0.3333333433, alpha: 1)
            case .rooster:
                self.label.text = most_probable_label
                self.view.backgroundColor = #colorLiteral(red: 0.9529411793, green: 0.6862745285, blue: 0.1333333403, alpha: 1)
            case .sneezing:
                self.label.text = most_probable_label
                self.view.backgroundColor = #colorLiteral(red: 0.3411764801, green: 0.6235294342, blue: 0.1686274558, alpha: 1)
            }
            print(count)
            if count >= 3 {
                t.invalidate()
            }
            count += 1
        }
    }
    
}

