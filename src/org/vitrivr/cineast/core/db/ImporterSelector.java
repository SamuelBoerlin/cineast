package org.vitrivr.cineast.core.db;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.vitrivr.cineast.core.config.Config;
import org.vitrivr.cineast.core.config.QueryConfig;
import org.vitrivr.cineast.core.config.ReadableQueryConfig;
import org.vitrivr.cineast.core.data.FixedSizePriorityQueue;
import org.vitrivr.cineast.core.data.distance.DistanceElement;
import org.vitrivr.cineast.core.data.providers.primitive.FloatTypeProvider;
import org.vitrivr.cineast.core.data.providers.primitive.PrimitiveTypeProvider;
import org.vitrivr.cineast.core.data.providers.primitive.ProviderDataType;
import org.vitrivr.cineast.core.importer.Importer;
import org.vitrivr.cineast.core.util.distance.BitSetComparator;
import org.vitrivr.cineast.core.util.distance.BitSetHammingDistance;
import org.vitrivr.cineast.core.util.distance.Distance;
import org.vitrivr.cineast.core.util.distance.FloatArrayDistance;
import org.vitrivr.cineast.core.util.distance.PrimitiveTypeMapDistanceComparator;

public abstract class ImporterSelector<T extends Importer<?>> implements DBSelector {

  private File file;
  private static final Logger LOGGER = LogManager.getLogger();

  @Override
  public boolean open(String name) {
    this.file = new File(
        Config.sharedConfig().getDatabase().getHost() + "/" + name + getFileExtension());
    return file.exists() && file.isFile() && file.canRead();
  }

  public boolean openFile(File file) {
    if (file == null) {
      throw new NullPointerException("file cannot be null");
    }
    this.file = file;
    return file.exists() && file.isFile() && file.canRead();
  }

  @Override
  public boolean close() {
    return true;
  }

  @Override
  public <E extends DistanceElement> List<E> getNearestNeighboursGeneric(int k,
      PrimitiveTypeProvider queryProvider, String column, Class<E> distanceElementClass,
      ReadableQueryConfig config) {
    List<Map<String, PrimitiveTypeProvider>> results;
    if (queryProvider.getType().equals(ProviderDataType.FLOAT_ARRAY) || queryProvider.getType()
        .equals(ProviderDataType.INT_ARRAY)) {
      results = getNearestNeighbourRows(k, queryProvider.getFloatArray(), column, config);
    } else {
     results = getNearestNeighbourRows(k, queryProvider, column, config);
    }
    return results.stream()
        .map(m -> DistanceElement.create(
            distanceElementClass, m.get("id").getString(), m.get("distance").getDouble()))
        .limit(k)
        .collect(Collectors.toList());
  }

  /**
   * Full table scan. Don't do it for performance-intensive stuff.
   */
  private List<Map<String, PrimitiveTypeProvider>> getNearestNeighbourRows(int k,
      PrimitiveTypeProvider queryProvider, String column, ReadableQueryConfig config) {
    if (queryProvider.getType().equals(ProviderDataType.FLOAT_ARRAY) || queryProvider.getType()
        .equals(ProviderDataType.INT_ARRAY)) {
      return getNearestNeighbourRows(k, PrimitiveTypeProvider.getSafeFloatArray(queryProvider),
          column, config);
    }
    LOGGER.debug("Switching to non-float based lookup, reading from file");
    Importer<?> importer = newImporter(this.file);

    Distance distance;
    FixedSizePriorityQueue<Map<String, PrimitiveTypeProvider>> knn;
    if (queryProvider.getType().equals(ProviderDataType.BITSET)) {
      distance = new BitSetHammingDistance();
      knn = FixedSizePriorityQueue
          .create(k, new BitSetComparator(column, distance, queryProvider.getBitSet()));
    } else {
      throw new RuntimeException(queryProvider.getType().toString());
    }

    Map<String, PrimitiveTypeProvider> map;
    while ((map = importer.readNextAsMap()) != null) {
      if (!map.containsKey(column)) {
        continue;
      }
      double d;
      if (queryProvider.getType().equals(ProviderDataType.BITSET)) {
        d = distance
            .applyAsDouble(queryProvider.getBitSet(), map.get(column).getBitSet());
        map.put("distance", new FloatTypeProvider((float) d));
        knn.add(map);
      } else {
        throw new RuntimeException(queryProvider.getType().toString());
      }
    }

    int len = Math.min(knn.size(), k);
    ArrayList<Map<String, PrimitiveTypeProvider>> _return = new ArrayList<>(len);

    for (Map<String, PrimitiveTypeProvider> i : knn) {
      _return.add(i);
      if (_return.size() >= len) {
        break;
      }
    }

    return _return;
  }


  @Override
  public List<Map<String, PrimitiveTypeProvider>> getNearestNeighbourRows(int k, float[] vector,
      String column, ReadableQueryConfig config) {

    config = QueryConfig.clone(config);

    Importer<?> importer = newImporter(this.file);

    FloatArrayDistance distance = FloatArrayDistance.fromQueryConfig(config);

    FixedSizePriorityQueue<Map<String, PrimitiveTypeProvider>> knn = FixedSizePriorityQueue
        .create(k, new PrimitiveTypeMapDistanceComparator(column, vector, distance));

    Map<String, PrimitiveTypeProvider> map;
    while ((map = importer.readNextAsMap()) != null) {
      if (!map.containsKey(column)) {
        continue;
      }
      double d = distance
          .applyAsDouble(vector, PrimitiveTypeProvider.getSafeFloatArray(map.get(column)));
      map.put("distance", new FloatTypeProvider((float) d));
      knn.add(map);
    }

    int len = Math.min(knn.size(), k);
    ArrayList<Map<String, PrimitiveTypeProvider>> _return = new ArrayList<>(len);

    for (Map<String, PrimitiveTypeProvider> i : knn) {
      _return.add(i);
      if (_return.size() >= len) {
        break;
      }
    }

    return _return;
  }

  @Override
  public List<float[]> getFeatureVectors(String fieldName, String value, String vectorName) {
    ArrayList<float[]> _return = new ArrayList<>(1);

    if (value == null || value.isEmpty()) {
      return _return;
    }

    Importer<?> importer = newImporter(this.file);
    Map<String, PrimitiveTypeProvider> map;
    while ((map = importer.readNextAsMap()) != null) {
      if (!map.containsKey(fieldName)) {
        continue;
      }
      if (!map.containsKey(vectorName)) {
        continue;
      }
      if (value.equals(map.get(fieldName).getString())) {
        _return.add(PrimitiveTypeProvider.getSafeFloatArray(map.get(vectorName)));
      }
    }

    return _return;
  }

  @Override
  public List<Map<String, PrimitiveTypeProvider>> getRows(String fieldName, String... values) {
    ArrayList<Map<String, PrimitiveTypeProvider>> _return = new ArrayList<>(1);

    if (values == null || values.length == 0) {
      return _return;
    }

    Importer<?> importer = newImporter(this.file);
    Map<String, PrimitiveTypeProvider> map;
    while ((map = importer.readNextAsMap()) != null) {
      if (!map.containsKey(fieldName)) {
        continue;
      }
      for (int i = 0; i < values.length; ++i) {
        if (values[i].equals(map.get(fieldName).getString())) {
          _return.add(map);
          break;
        }
      }

    }

    return _return;
  }

  @Override
  public List<Map<String, PrimitiveTypeProvider>> getRows(String fieldName,
      Iterable<String> values) {
    if (values == null) {
      return new ArrayList<>(0);
    }

    ArrayList<String> tmp = new ArrayList<>();
    for (String value : values) {
      tmp.add(value);
    }

    String[] valueArr = new String[tmp.size()];
    tmp.toArray(valueArr);

    return this.getRows(fieldName, valueArr);
  }

  @Override
  public List<PrimitiveTypeProvider> getAll(String column) {
    List<PrimitiveTypeProvider> _return = new ArrayList<>();

    Importer<?> importer = newImporter(this.file);
    Map<String, PrimitiveTypeProvider> map;
    while ((map = importer.readNextAsMap()) != null) {
      PrimitiveTypeProvider p = map.get(column);
      if (p != null) {
        _return.add(p);
      }
    }
    return _return;
  }

  @Override
  public List<Map<String, PrimitiveTypeProvider>> getAll() {

    List<Map<String, PrimitiveTypeProvider>> _return = new ArrayList<>();

    Importer<?> importer = newImporter(this.file);
    Map<String, PrimitiveTypeProvider> map;
    while ((map = importer.readNextAsMap()) != null) {
      _return.add(map);
    }
    return _return;
  }

  @Override
  public boolean existsEntity(String name) {
    File file = new File(
        Config.sharedConfig().getDatabase().getHost() + "/" + name + getFileExtension());
    return file.exists() && file.isFile() && file.canRead();
  }

  protected abstract T newImporter(File f);

  protected abstract String getFileExtension();

  @Override
  public <T extends DistanceElement> List<T> getBatchedNearestNeighbours(int k,
      List<float[]> vectors, String column, Class<T> distanceElementClass,
      List<ReadableQueryConfig> configs) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public <T extends DistanceElement> List<T> getCombinedNearestNeighbours(int k,
      List<float[]> vectors, String column, Class<T> distanceElementClass,
      List<ReadableQueryConfig> configs, MergeOperation merge, Map<String, String> options) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public List<Map<String, PrimitiveTypeProvider>> getRows(String fieldName,
      RelationalOperator operator, Iterable<String> values) {
    throw new IllegalStateException("Not implemented.");
  }

  public List<Map<String, PrimitiveTypeProvider>> getFulltextRows(int rows, String fieldname,
      String... terms) {
    throw new IllegalStateException("Not implemented.");
  }

  @Override
  public boolean ping() {
    return true;
  }
}
