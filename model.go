package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"

	vision "cloud.google.com/go/vision/apiv1"
	"github.com/disintegration/imaging"
)

const (
	SourcePath = "E:/temp_uploads/Photos/Fruit/"
	FoodType   = "Fruit" // "Change as needed"
)

func loadFoodNames(foodType string) ([]string, error) {
	filePath := fmt.Sprintf("dict/%s.dict", foodType)
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(data), "\n")
	for i := range lines {
		lines[i] = strings.TrimSpace(strings.ToLower(lines[i]))
	}
	return lines, nil
}

func recognizeFood(imagePath string, foodList []string) error {
	startTime := time.Now()

	// Open and resize the image using imaging
	img, err := imaging.Open(imagePath)
	if err != nil {
		return err
	}
	width := 800
	height := int(float64(img.Bounds().Dy()) * (800.0 / float64(img.Bounds().Dx())))
	resizedImg := imaging.Resize(img, width, height, imaging.Lanczos)

	// Save the resized image
	outputPath := SourcePath + "output.jpg"
	err = imaging.Save(resizedImg, outputPath)
	if err != nil {
		return err
	}

	// Create a Vision API client
	ctx := context.Background()
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Read the image file
	content, err := ioutil.ReadFile(outputPath)
	if err != nil {
		return err
	}

	// Perform label detection
	image := vision.NewImageFromBytes(content)
	annotations, err := client.DetectLabels(ctx, image, nil, 10)
	if err != nil {
		return err
	}

	for _, annotation := range annotations {
		desc := strings.ToLower(annotation.Description)
		score := annotation.Score
		fmt.Printf("Label: %s  Score: %.2f\n", desc, score)
		for _, food := range foodList {
			if desc == food {
				fmt.Printf("Recognized food: %s with score: %.2f\n", desc, score)
				break
			}
		}
	}

	fmt.Printf("Total time: %v\n", time.Since(startTime))
	return nil
}

func main() {
	fmt.Println("---------- Start FOOD Recognition --------")

	// Load food names
	foodList, err := loadFoodNames(FoodType)
	if err != nil {
		log.Fatalf("Failed to load food names: %v", err)
	}
	fmt.Println("Food List:", foodList)

	// Recognize food
	imagePath := SourcePath + "1.jpg"
	err = recognizeFood(imagePath, foodList)
	if err != nil {
		log.Fatalf("Failed to recognize food: %v", err)
	}

	fmt.Println("---------- End ----------")
}