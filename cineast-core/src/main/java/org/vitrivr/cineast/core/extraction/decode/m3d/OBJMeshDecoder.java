package org.vitrivr.cineast.core.extraction.decode.m3d;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.io.FilenameUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector3i;
import org.joml.Vector4i;
import org.vitrivr.cineast.core.config.CacheConfig;
import org.vitrivr.cineast.core.config.DecoderConfig;
import org.vitrivr.cineast.core.data.m3d.Mesh;
import org.vitrivr.cineast.core.data.m3d.Mesh.Vertex;
import org.vitrivr.cineast.core.extraction.decode.general.Decoder;
import org.vitrivr.cineast.core.extraction.decode.image.DefaultImageDecoder;
import org.vitrivr.cineast.core.util.LogHelper;
import org.vitrivr.cineast.core.util.MimeTypeHelper;

/**
 * Decodes Wavefront OBJ (.obj) files and returns a Mesh representation. Requires JOML to work properly.
 *
 * Texture information is currently discarded!
 *
 * @author rgasser
 * @version 1.1
 */
public class OBJMeshDecoder implements Decoder<Mesh> {
    /** Default logging facility. */
    private static final Logger LOGGER = LogManager.getLogger();

    /** HashSet containing all the mime-types supported by this ImageDecoder instance. */
    private static final Set<String> supportedFiles;
    static {
        HashSet<String> tmp = new HashSet<>();
        tmp.add("application/3d-obj");
        supportedFiles = Collections.unmodifiableSet(tmp);
    }

    /** Path to the input file. */
    private Path inputFile;

    /** Flag indicating whether or not the Decoder is done decoding and the content has been obtained. */
    private AtomicBoolean complete = new AtomicBoolean(false);

    /**
     * Initializes the decoder with a file. This is a necessary step before content can be retrieved from
     * the decoder by means of the getNext() method.
     *
     * @param path Path to the file that should be decoded.
     * @param decoderConfig {@link DecoderConfig} used by this {@link Decoder}.
     * @param cacheConfig The {@link CacheConfig} used by this {@link Decoder}
     * @return True if initialization was successful, false otherwise.
     */
    @Override
    public boolean init(Path path, DecoderConfig decoderConfig, CacheConfig cacheConfig) {
        this.inputFile = path;
        this.complete.set(false);
        return true;
    }

    /**
     * Fetches the next piece of content of type T and returns it. This method can be safely invoked until
     * complete() returns false. From which on this method will return null.
     *
     * @return Content of type T.
     */
    @Override
    public Mesh getNext() {
        Mesh mesh = new Mesh(100,100);
        try {
            InputStream is = Files.newInputStream(this.inputFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            String line = null;
            
            List<Vector3f> vertices = new ArrayList<>();
            List<Vector2f> uvs = new ArrayList<>();
            List<Vector3f> normals = new ArrayList<>();
            
            Map<Vertex, Integer> verticesMap = new HashMap<>();
            
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split("\\s+");
                switch (tokens[0]) {
                    case "v":
                    	vertices.add(new Vector3f(Float.parseFloat(tokens[1]), Float.parseFloat(tokens[2]), Float.parseFloat(tokens[3])));
                        break;
                    case "vt":
                    	uvs.add(new Vector2f(Float.parseFloat(tokens[1]), Float.parseFloat(tokens[2])));
                    	break;
                    case "vn":
                    	normals.add(new Vector3f(Float.parseFloat(tokens[1]), Float.parseFloat(tokens[2]), Float.parseFloat(tokens[3])));
                        break;
                    case "f":
                    	boolean quad = (tokens.length == 5);
                    	
                    	if(!tokens[1].contains("/")) {
                        	//Face contains just vertices
                    		int i1 = this.createVertex(mesh, verticesMap, vertices, null, null, null, Integer.parseInt(tokens[1]) - 1, -1, -1, -1);
                    		int i2 = this.createVertex(mesh, verticesMap, vertices, null, null, null, Integer.parseInt(tokens[2]) - 1, -1, -1, -1);
                    		int i3 = this.createVertex(mesh, verticesMap, vertices, null, null, null, Integer.parseInt(tokens[3]) - 1, -1, -1, -1);
                    		if(quad) {
                        		int i4 = this.createVertex(mesh, verticesMap, vertices, null, null, null, Integer.parseInt(tokens[4]) - 1, -1, -1, -1);
                    			mesh.addFace(new Vector4i(i1, i2, i3, i4));
                    		} else {
                    			mesh.addFace(new Vector3i(i1, i2, i3));
                    		}
                        } else if(tokens[2].contains("//")) {
                        	//Face contains vertices and normals
                        	String[] p1 = tokens[1].split("//");
                            String[] p2 = tokens[2].split("//");
                            String[] p3 = tokens[3].split("//");
                        	int i1 = this.createVertex(mesh, verticesMap, vertices, null, normals, null, Integer.parseInt(p1[0]) - 1, -1, Integer.parseInt(p1[1]) - 1, -1);
                        	int i2 = this.createVertex(mesh, verticesMap, vertices, null, normals, null, Integer.parseInt(p2[0]) - 1, -1, Integer.parseInt(p2[1]) - 1, -1);
                        	int i3 = this.createVertex(mesh, verticesMap, vertices, null, normals, null, Integer.parseInt(p3[0]) - 1, -1, Integer.parseInt(p3[1]) - 1, -1);
                        	if(quad) {
                        		String[] p4 = tokens[4].split("//");
                        		int i4 = this.createVertex(mesh, verticesMap, vertices, null, normals, null, Integer.parseInt(p4[0]) - 1, -1, Integer.parseInt(p4[1]) - 1, -1);
                        		mesh.addFace(new Vector4i(i1, i2, i3, i4));
                    		} else {
                    			mesh.addFace(new Vector3i(i1, i2, i3));
                    		}
                        } else {
                        	//Face contains vertices, normals and UVs
                        	String[] p1 = tokens[1].split("/");
                            String[] p2 = tokens[2].split("/");
                            String[] p3 = tokens[3].split("/");
                        	int i1 = this.createVertex(mesh, verticesMap, vertices, null, normals, uvs, Integer.parseInt(p1[0]) - 1, -1, Integer.parseInt(p1[2]) - 1, Integer.parseInt(p1[1]) - 1);
                        	int i2 = this.createVertex(mesh, verticesMap, vertices, null, normals, uvs, Integer.parseInt(p2[0]) - 1, -1, Integer.parseInt(p2[2]) - 1, Integer.parseInt(p2[1]) - 1);
                        	int i3 = this.createVertex(mesh, verticesMap, vertices, null, normals, uvs, Integer.parseInt(p3[0]) - 1, -1, Integer.parseInt(p3[2]) - 1, Integer.parseInt(p3[1]) - 1);
                        	if(quad) {
                        		String[] p4 = tokens[4].split("/");
                        		int i4 = this.createVertex(mesh, verticesMap, vertices, null, normals, uvs, Integer.parseInt(p4[0]) - 1, -1, Integer.parseInt(p4[2]) - 1, Integer.parseInt(p4[1]) - 1);
                        		mesh.addFace(new Vector4i(i1, i2, i3, i4));
                    		} else {
                    			mesh.addFace(new Vector3i(i1, i2, i3));
                    		}
                        }
                        break;
                    default:
                        break;
                }
            }

            br.close(); /* Closes the input stream. */
        } catch (IOException e) {
            LOGGER.error("Could not decode OBJ file {} due to an IO exception ({})", this.inputFile.toString(), LogHelper.getStackTrace(e));
            mesh = null;
        } catch (NumberFormatException e) {
            LOGGER.error("Could not decode OBJ file {} because one of the tokens could not be converted to a valid number.", this.inputFile.toString());
            mesh = null;
        } catch (ArrayIndexOutOfBoundsException e) {
            LOGGER.error("Could not decode OBJ file {} because one of the faces points to invalid vertex indices.", this.inputFile.toString());
            mesh = null;
        } finally {
            this.complete.set(true);
        }
        
        //Load texture
        this.loadTexture(mesh);
        
        return mesh;
    }
    
    /**
     * Loads the texture for the given mesh and inputFile.
     * 
     * @param mesh
     */
    private void loadTexture(Mesh mesh) {
    	//TODO Use OBJ's MTL to find texture file
    	
    	final String modelFileName = FilenameUtils.removeExtension(this.inputFile.getFileName().toString());
    	
    	Path texturePath = null;
    	BufferedImage texture = null;
    	
        try(DefaultImageDecoder imageDecoder = new DefaultImageDecoder()) {
	        try(Stream<Path> paths = Files.walk(this.inputFile.getParent())) {
	        	List<Path> similarFiles = paths.filter(path -> modelFileName.equals(FilenameUtils.removeExtension(path.getFileName().toString()))).collect(Collectors.toList());
	        	
	        	for(Path similarFile : similarFiles) {
	        		String mimeType = MimeTypeHelper.getContentType(similarFile);
	        		
	        		if(imageDecoder.supportedFiles().contains(mimeType)) {
	        			imageDecoder.init(similarFile, null, null);
	        			
	        			while(!imageDecoder.complete()) {
	        				texture = imageDecoder.getNext();
	        			}
	        			
	        			if(texture != null) {
	        				texturePath = similarFile;
	        				break;
	        			}
	        		}
	        	}
	        } catch (IOException e) {
				e.printStackTrace();
				LOGGER.error("Could not load textures for OBJ file {} due to an IO exception ({})", this.inputFile.toString(), LogHelper.getStackTrace(e));
			}
        }
        
        if(texture != null) {
        	mesh.setTexture(texture);
        }
        
        if(texturePath != null) {
        	mesh.setTexturePath(texturePath.toString());
        }
    }
    
    /**
     * Creates a new vertex with the given attributes and returns the vertex's index.
     * Deduplicates identical vertices.
     * @param mesh
     * @param verticesMap
     * @param vertices
     * @param colors
     * @param normals
     * @param uvs
     * @param vi
     * @param ci
     * @param ni
     * @param ti
     * @return
     */
    private int createVertex(Mesh mesh, Map<Vertex, Integer> verticesMap, List<Vector3f> vertices, List<Vector3f> colors, List<Vector3f> normals, List<Vector2f> uvs, int vi, int ci, int ni, int ti) {
    	//TODO Logger warnings for out of bounds indices
    	
    	Vector3f position = vi >= 0 ? vertices.get(vi) : new Vector3f(0.0f, 0.0f, 0.0f);
    	Vector3f color = ci >= 0 ? colors.get(ci) : new Vector3f(1.0f, 1.0f, 1.0f);
    	Vector3f normal = ni >= 0 ? normals.get(ni) : new Vector3f(0.0f, 0.0f, 0.0f);
    	Vector2f uv = ti >= 0 ? uvs.get(ti) : new Vector2f(0.0f, 0.0f);
    	
    	Vertex vertex = mesh.new Vertex(position, color, normal, uv);
    	
    	Integer assignedIndex = verticesMap.get(vertex);
    	if(assignedIndex != null) {
    		return assignedIndex;
    	} else {
    		int newIndex = verticesMap.size();
    		verticesMap.put(vertex, newIndex);
    		mesh.addVertex(position, color, normal, uv);
    		return newIndex;
    	}
    }

    /**
     * Returns the total number of content pieces T this decoder can return for a given file.
     *
     * @return
     */
    @Override
    public int count() {
        return 1;
    }

    /**
     * Closes the Decoder. This method should cleanup and relinquish all resources.
     * <p>
     * Note: It is unsafe to re-use a Decoder after it has been closed.
     */
    @Override
    public void close() {}

    /**
     * Indicates whether or not a particular instance of the Decoder interface can
     * be re-used or not. This property can be leveraged to reduce the memory-footpring
     * of the application.
     *
     * @return True if re-use is possible, false otherwise.
     */
    @Override
    public boolean canBeReused() {
        return true;
    }

    /**
     * Indicates whether or not the current decoder instance is complete i.e. if there is
     * content left that can be obtained.
     *
     * @return true if there is still content, false otherwise.
     */
    @Override
    public boolean complete() {
        return this.complete.get();
    }

    /**
     * Returns a set of the mime/types of supported files.
     *
     * @return Set of the mime-type of file formats that are supported by the current Decoder instance.
     */
    @Override
    public Set<String> supportedFiles() {
        return supportedFiles;
    }
}
