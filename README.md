[![Build Status](https://travis-ci.org/vitrivr/cineast.svg?branch=master)](https://travis-ci.org/vitrivr/cineast)

# Cineast
Cineast is a multi-feature content-based mulitmedia retrieval engine. It is capable of retrieving images, audio- and video sequences as well as 3d models based on edge or color sketches, sketch-based motion queries and example objects.
Cineast is written in Java and uses [ADAMpro](https://github.com/vitrivr/ADAMpro) as a storage backend.

## Building Cineast
Cineast can be built using [Gradle](http://gradle.org/). Building and running it is as easy as
```
 git clone --recursive https://github.com/vitrivr/cineast.git cineast
 cd cineast
 ./gradlew deploy
 cd build/libs
 java -jar cineast.jar
 ```

For more setup information, consult our [Wiki](https://github.com/vitrivr/cineast/wiki)

## Prerequisites
### System dependencies
* git
* JDK 8 or higher
* Optionally: You should install [Docker](https://docs.docker.com/engine/installation/linux/ubuntulinux/) which you can use for ADAMpro.

### 3D rendering
For 3D rendering (required in order to support 3D models) you either need a video card or Mesa 3D. The JOGL library supports both. Rendering on Headless devices has been successfully tested with Xvfb. The following steps are required to enable
3D rendering support on a headless device without video card (Ubuntu 16.04.1 LTS)

1. Install Mesa 3D (should come pre-installed on Ubuntu). Check with `dpkg -l | grep mesa`
2. Install Xvfb:

 ```
 sudo apt-get install xvfb
 ```
 
3. Start a new screen:

 ```
 sudo Xvfb :1 -ac -screen 0 1024x768x24 &
 ```
 
4. Using the new screen, start Cineast:

 ```
 DISPLAY=:1 java -jar cineast.jar -3d
 ```
 
The -3d option will perform a 3D test. If it succeeds, cineast should generate a PNG image depicting two coloured
triangles on a black background.

## Contribution

> Contributions are always welcome.

### Versioning

Cineast uses [semantic versioning](https://semver.org). See [the releases page](https://github.com/vitrivr/cineast/releases).

### Code Style

Cineast primarily uses the [Google Java Styleguide](https://google.github.io/styleguide/javaguide.html).
Please keep the changes you do in compliance with it.

To automatically apply the styleguide in [IntelliJ IDEA](https://www.jetbrains.com/idea/) import the [styleguide](https://github.com/google/styleguide/blob/gh-pages/intellij-java-google-style.xml) in _File_ -> _Settings_ -> _Editor_ -> _Code Style_ -> _Java_ and import the [scheme](https://github.com/google/styleguide/blob/gh-pages/intellij-java-google-style.xml) via the gear icon.

You can also use [Eclipse](https://www.eclipse.org/) for development and use Google's [styleguide for eclipse](https://github.com/google/styleguide/blob/gh-pages/eclipse-java-google-style.xml).
