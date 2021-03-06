package org.vitrivr.cineast.api.websocket.handlers.queries;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.eclipse.jetty.websocket.api.Session;
import org.vitrivr.cineast.api.websocket.handlers.abstracts.StatelessWebsocketMessageHandler;
import org.vitrivr.cineast.core.config.Config;
import org.vitrivr.cineast.core.config.QueryConfig;
import org.vitrivr.cineast.core.data.entities.MediaObjectDescriptor;
import org.vitrivr.cineast.core.data.entities.MediaObjectMetadataDescriptor;
import org.vitrivr.cineast.core.data.entities.MediaSegmentDescriptor;
import org.vitrivr.cineast.core.data.entities.MediaSegmentMetadataDescriptor;
import org.vitrivr.cineast.core.data.messages.query.Query;
import org.vitrivr.cineast.core.data.messages.result.MediaObjectMetadataQueryResult;
import org.vitrivr.cineast.core.data.messages.result.MediaSegmentMetadataQueryResult;
import org.vitrivr.cineast.core.data.messages.result.QueryEnd;
import org.vitrivr.cineast.core.data.messages.result.QueryError;
import org.vitrivr.cineast.core.data.messages.result.QueryStart;
import org.vitrivr.cineast.core.db.dao.reader.MediaObjectMetadataReader;
import org.vitrivr.cineast.core.db.dao.reader.MediaObjectReader;
import org.vitrivr.cineast.core.db.dao.reader.MediaSegmentMetadataReader;
import org.vitrivr.cineast.core.db.dao.reader.MediaSegmentReader;
import org.vitrivr.cineast.core.util.LogHelper;

/**
 * @author rgasser
 * @version 1.0
 * @created 27.04.17
 */
public abstract class AbstractQueryMessageHandler<T extends Query> extends StatelessWebsocketMessageHandler<T> {

  private static final Logger LOGGER = LogManager.getLogger();

  /**
   * {@link MediaSegmentReader} instance used to read segments from the storage layer.
   */
  protected final MediaSegmentReader mediaSegmentReader = new MediaSegmentReader();

  /**
   * {@link MediaObjectReader} instance used to read multimedia objects from the storage layer.
   */
  private final MediaObjectReader mediaObjectReader = new MediaObjectReader();

  /**
   * {@link MediaSegmentMetadataReader} instance used to read {@link MediaSegmentMetadataDescriptor}s from the storage layer.
   */
  private final MediaSegmentMetadataReader segmentMetadataReader = new MediaSegmentMetadataReader();

  /**
   * {@link MediaObjectMetadataReader} instance used to read {@link MediaObjectMetadataReader}s from the storage layer.
   */
  private final MediaObjectMetadataReader objectMetadataReader = new MediaObjectMetadataReader();

  /**
   * Handles a {@link Query} message
   *
   * @param session WebSocketSession for which the message arrived.
   * @param message Message of type a that needs to be handled.
   */
  public final void handle(Session session, T message) {
    try {
      final QueryConfig qconf = (message.getQueryConfig() == null) ? QueryConfig.newQueryConfigFromOther(Config.sharedConfig().getQuery()) : message.getQueryConfig();
      final String uuid = qconf.getQueryId().toString();

      try {
        /* Begin of Query: Send QueryStart Message to Client. */
        this.write(session, new QueryStart(uuid));
        LOGGER.trace("Executing query from message {}", message);

        /* Execute actual query. */
        this.execute(session, qconf, message);
      } catch (Exception e) {
        /* Error: Send QueryError Message to Client. */
        LOGGER.error("An exception occurred during execution of similarity query message {}.", LogHelper.getStackTrace(e));
        this.write(session, new QueryError(uuid, e.getMessage()));
        return;
      }

      /* End of Query: Send QueryEnd Message to Client. */
      this.write(session, new QueryEnd(uuid));
    } catch (Throwable t) {
      t.printStackTrace();
    }
  }

  /**
   * Executes the actual query specified by the {@link Query} object.
   *
   * @param session WebSocket session the invocation is associated with.
   * @param qconf The {@link QueryConfig} that contains additional specifications.
   * @param message {@link Query} that must be executed.
   */
  protected abstract void execute(Session session, QueryConfig qconf, T message) throws Exception;

  /**
   * Performs a lookup for the {@link MediaSegmentDescriptor} identified by the provided IDs and returns a list of the {@link MediaSegmentDescriptor}s that were found.
   *
   * @param segmentIds List of segment IDs that should be looked up.
   * @return List of found {@link MediaSegmentDescriptor}
   */
  protected List<MediaSegmentDescriptor> loadSegments(List<String> segmentIds) {
    final Map<String, MediaSegmentDescriptor> map = this.mediaSegmentReader.lookUpSegments(segmentIds);
    final ArrayList<MediaSegmentDescriptor> sdList = new ArrayList<>(map.size());
    segmentIds.stream().filter(map::containsKey).forEach(s -> sdList.add(map.get(s)));
    return sdList;
  }

  /**
   * Performs a lookup for the {@link MediaObjectDescriptor} identified by the provided object IDs and returns a list of the {@link MediaSegmentDescriptor}s that were found.
   *
   * @param objectIds List of object IDs that should be looked up.
   * @return List of found {@link MediaObjectDescriptor}
   */
  protected List<MediaObjectDescriptor> loadObjects(List<String> objectIds) {
    final Map<String, MediaObjectDescriptor> map = this.mediaObjectReader.lookUpObjects(objectIds);
    final ArrayList<MediaObjectDescriptor> vdList = new ArrayList<>(map.size());
    objectIds.stream().filter(map::containsKey).forEach(s -> vdList.add(map.get(s)));
    return vdList;
  }

  /**
   * Performs a lookup for {@link MediaObjectMetadataReader} identified by the provided segment IDs.
   *
   * @param session The WebSocket session to write the data to.
   * @param queryId The current query id used for transmitting data back.
   * @param objectIds List of object IDs for which to lookup metadata.
   */
  protected void loadAndWriteObjectMetadata(Session session, String queryId, List<String> objectIds) {
    final List<MediaObjectMetadataDescriptor> objectMetadata = this.objectMetadataReader.lookupMultimediaMetadata(objectIds);
    if (!objectMetadata.isEmpty()) {
      this.write(session, new MediaObjectMetadataQueryResult(queryId, objectMetadata));
    }
  }

  /**
   * Performs a lookup for {@link MediaSegmentMetadataDescriptor} identified by the provided segment IDs and writes them to the WebSocket stream.
   *
   * @param session The WebSocket session to write the data to.
   * @param queryId The current query id used for transmitting data back.
   * @param segmentIds List of segment IDs for which to lookup metadata.
   */
  protected void loadAndWriteSegmentMetadata(Session session, String queryId, List<String> segmentIds) {
    final List<MediaSegmentMetadataDescriptor> segmentMetadata = this.segmentMetadataReader.lookupMultimediaMetadata(segmentIds);
    if (!segmentMetadata.isEmpty()) {
      this.write(session, new MediaSegmentMetadataQueryResult(queryId, segmentMetadata));
    }
  }
}
