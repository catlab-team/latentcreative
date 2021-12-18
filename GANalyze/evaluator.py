import numpy as np
import matplotlib.pyplot as plt
from property_evaluation.functions import *
import os

class MeanAssessorScoreEvaluator():
    def __init__(self, alpha_values=[-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3]):
        
        self.scores = {}
        self.colorfulness = {} 
        self.brightness = {}
        self.redness = {}
        self.entropy = {}

        self.centerness = {}
        self.squareness = {}
        self.object_size = {}

        #self.colorfulness_min = 9999
        #self.colorfulness_max = -9999
        #self.entropy_min = 9999
        #self.entropy_max = -9999

        self.rcnn = MaskRCNN()

        self.alpha_values = []
        for alpha in alpha_values:
            self.alpha_values.append("{:.3f}".format(alpha))
            self.scores["{:.3f}".format(alpha)] = []
            self.colorfulness["{:.3f}".format(alpha)] = []
            self.brightness["{:.3f}".format(alpha)] = []
            self.redness["{:.3f}".format(alpha)] = []
            self.entropy["{:.3f}".format(alpha)] = []

            self.centerness["{:.3f}".format(alpha)] = []
            self.squareness["{:.3f}".format(alpha)] = []
            self.object_size["{:.3f}".format(alpha)] = []

    def update(self, alpha, outscores_mean, images=None):
        alpha_str = "{:.3f}".format(alpha)
        if alpha_str not in self.alpha_values:
            assert False, "set alpha values correctly"
        self.scores[alpha_str].append(outscores_mean)
        if images is not None:
            for img in images:
                colorfulness_score, brightness_score, redness_score, entropy_score, \
                centerness_score, object_size_score, squareness_score = \
                                                    calculate_features(img, self.rcnn)

                self.colorfulness[alpha_str].append(colorfulness_score)
                self.brightness[alpha_str].append(brightness_score)
                self.redness[alpha_str].append(redness_score)
                self.entropy[alpha_str].append(entropy_score)
                #self.colorfulness_min = min(self.colorfulness_min, colorfulness_score)
                #self.colorfulness_max = max(self.colorfulness_max, colorfulness_score)
                #self.entropy_min = min(self.entropy_min, entropy_score)
                #self.entropy_max = max(self.entropy_max, entropy_score)

                if centerness_score:
                    self.centerness[alpha_str].append(centerness_score)
                    self.object_size[alpha_str].append(object_size_score)
                    self.squareness[alpha_str].append(squareness_score)


    def normalize(self, array):
        assert (len(array)%2) != 0
        mid_element = array[len(array)//2]
        normalized = []
        for e in array:
            normalized.append((e-mid_element)/mid_element)
        return normalized

    def finish(self, plot_path, checkpoint_dir, way, is_rcnn=False):

        alphas = []
        score_means = []
        score_means_std = []
        redness_means = []
        entropy_means = []
        colorfulness_means = []
        brightness_means = []

        squareness_means = []
        object_size_means = []
        centerness_means = []

        for alpha in self.scores:
            print("Alpha: {}, Score Mean: {:.3f}, Score Std: {:.3f}".\
                format(alpha, np.mean(self.scores[alpha]), np.std(self.scores[alpha])))
            alphas.append(float(alpha))
            score_means.append(np.mean(self.scores[alpha]))
            score_means_std.append(np.std(self.scores[alpha]))
        
        
        if len(self.colorfulness) != 0:
            for alpha in self.scores:
                redness_score = np.mean(self.redness[alpha])
                print("Redness: {}, Score Mean: {:.3f}".format(alpha, redness_score))
                redness_means.append(redness_score)

            for alpha in self.scores:
                colorfulness_score = np.mean(self.colorfulness[alpha])
                print("Colorfulness: {}, Score Mean: {:.3f}".format(alpha, colorfulness_score))
                colorfulness_means.append(colorfulness_score)

            for alpha in self.scores:
                brightness_score = np.mean(self.brightness[alpha])
                print("Brightness: {}, Score Mean: {:.3f}".format(alpha, brightness_score))
                brightness_means.append(brightness_score)

            for alpha in self.scores:
                entropy_score = np.mean(self.entropy[alpha])
                print("Entropy: {}, Score Mean: {:.3f}".format(alpha, entropy_score))
                entropy_means.append(entropy_score)

            if len(self.squareness) != 0:
                for alpha in self.scores:
                    squareness_score = np.mean(self.squareness[alpha])
                    print("Squareness: {}, Score Mean: {:.3f}".format(alpha, squareness_score))
                    squareness_means.append(squareness_score)

                for alpha in self.scores:
                    object_size_score = np.mean(self.object_size[alpha])
                    print("Object_size: {}, Score Mean: {:.3f}".format(alpha, object_size_score))
                    object_size_means.append(object_size_score)

                for alpha in self.scores:
                    centerness_score = np.mean(self.centerness[alpha])
                    print("Centerness: {}, Score Mean: {:.3f}".format(alpha, centerness_score))
                    centerness_means.append(centerness_score)
        
        redness_means = self.normalize(redness_means)
        entropy_means = self.normalize(entropy_means)
        colorfulness_means = self.normalize(colorfulness_means)
        brightness_means = self.normalize(brightness_means)

        if len(self.colorfulness) != 0:
            squareness_means = self.normalize(squareness_means)
            object_size_means = self.normalize(object_size_means)
            centerness_means = self.normalize(centerness_means)

        file_name = "alphas_rcnn" if is_rcnn else "alphas"  
        if way == -1:
            file_name = os.path.join(checkpoint_dir, "{}.txt".format(file_name))
        else:
            file_name = os.path.join(checkpoint_dir, file_name+str(way)+".txt")
        with open(file_name, "w") as file:
            file.write("alphas = [")
            for i in range(len(alphas)-1):
                file.write(str(alphas[i])+", ")
            file.write(str(alphas[-1]))  
            file.write("]\n")
            
            file.write("x = [")
            for i in range(len(score_means)-1):
                file.write(str(score_means[i])+", ")
            file.write(str(score_means[-1]))  
            file.write("]\n")
            
            file.write("x_std = [")
            for i in range(len(score_means_std)-1):
                file.write(str(score_means_std[i])+", ")
            file.write(str(score_means_std[-1]))  
            file.write("]\n")
            
            file.write("redness = [")
            for i in range(len(redness_means)-1):
                file.write(str(redness_means[i])+", ")
            file.write(str(redness_means[-1]))  
            file.write("]\n")
            
            file.write("colorfulness = [")
            for i in range(len(colorfulness_means)-1):
                file.write(str(colorfulness_means[i])+", ")
            file.write(str(colorfulness_means[-1]))  
            file.write("]\n")
            
            file.write("brightness = [")
            for i in range(len(brightness_means)-1):
                file.write(str(brightness_means[i])+", ")
            file.write(str(brightness_means[-1]))  
            file.write("]\n")
            
            file.write("entropy = [")
            for i in range(len(entropy_means)-1):
                file.write(str(entropy_means[i])+", ")
            file.write(str(entropy_means[-1]))  
            file.write("]\n")

            if len(self.squareness) != 0:
                file.write("squareness = [")
                for i in range(len(squareness_means)-1):
                    file.write(str(squareness_means[i])+", ")
                file.write(str(squareness_means[-1]))  
                file.write("]\n")

                file.write("centerness = [")
                for i in range(len(centerness_means)-1):
                    file.write(str(centerness_means[i])+", ")
                file.write(str(centerness_means[-1]))  
                file.write("]\n")

                file.write("object_size = [")
                for i in range(len(object_size_means)-1):
                    file.write(str(object_size_means[i])+", ")
                file.write(str(object_size_means[-1]))  
                file.write("]\n")
                    
        if plot_path is None:
            return
        
        fig = plt.figure()

        if len(self.colorfulness) != 0:
            
            axis1 = fig.add_subplot(331)
            axis1.plot(alphas, score_means, 'xb-')
            axis1.set_title("Assessor Score")

            axis2 = fig.add_subplot(332)
            axis2.plot(alphas, redness_means, 'xb-')
            axis2.set_title("Redness Score")
            
            axis3 = fig.add_subplot(333)
            axis3.plot(alphas, entropy_means, 'xb-')
            axis3.set_title("Entropy Score")

            axis4 = fig.add_subplot(334)
            axis4.plot(alphas, brightness_means, 'xb-')
            axis4.set_title("Brightness Score")

            axis5 = fig.add_subplot(335)
            axis5.plot(alphas, colorfulness_means, 'xb-')
            axis5.set_title("Colorfulness Score")

            axis6 = fig.add_subplot(336)
            axis6.plot(alphas, centerness_means, 'xb-')
            axis6.set_title("Centerness Score")

            axis7 = fig.add_subplot(337)
            axis7.plot(alphas, squareness_means, 'xb-')
            axis7.set_title("Squareness Score")

            axis8 = fig.add_subplot(338)
            axis8.plot(alphas, object_size_means, 'xb-')
            axis8.set_title("Object Size Score")
        
        else:
            plt.plot(alphas, score_means, 'xb-')
        
        plt.savefig(str(way)+plot_path)
