# Real-Time Capture Images

This folder contains real microscopy/capture images that will be displayed in the picture box during simulation.

## How to Add Images

1. Add your images to this folder (JPG, PNG, GIF, WEBP formats supported)
2. Update the `index.html` file to load them

## Example

Add your files:
```
images/
  ├── microplastic1.jpg
  ├── microplastic2.jpg
  ├── capture_microscopy.png
  └── xenobot_capture.jpg
```

Then in `index.html` (around line 1030), add:
```javascript
loadRealPictures([
  'images/microplastic1.jpg',
  'images/microplastic2.jpg',
  'images/capture_microscopy.png',
  'images/xenobot_capture.jpg'
]);
```

## Tips

- Use high-quality microscopy images for best results
- Images will be cropped to fit the 180x180px box (centered)
- Pictures cycle randomly every 2 seconds during simulation
- Supported formats: .jpg, .jpeg, .png, .gif, .webp
